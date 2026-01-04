import os
import glob
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# =========================
# 0) Reproducibility
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1) Data Loading (batch*.dat)
# =========================
def load_batch_dat(path: str):
    """
    파일 포맷(예시):
      1;10.000000 1:15596.162100 2:1.868245 ... 128:-2.654529
    - 첫 토큰: gas_id;concentration
    - 이후 128개 토큰: idx:value
    """
    df = pd.read_csv(path, sep=r"\s+|,", header=None, engine="python", dtype=str)
    # gas label (1~6) -> 0~5
    y = df[0].str.split(";").str[0].astype(int).to_numpy() - 1

    # 128 features: columns 1..128, each "k:value"
    feats = []
    for c in range(1, 129):
        vals = df[c].str.split(":").str[1].astype(float).to_numpy()
        feats.append(vals)
    X = np.stack(feats, axis=1).astype(np.float32)
    return X, y.astype(np.int64)


def load_all_batches(data_dir: str):
    paths = sorted(glob.glob(os.path.join(data_dir, "batch*.dat")))
    if len(paths) == 0:
        raise FileNotFoundError("batch*.dat files not found in data_dir")

    X_list, y_list, d_list = [], [], []
    for p in paths:
        # batch number from filename: batch10.dat -> 10
        bn = int(os.path.splitext(os.path.basename(p))[0].replace("batch", ""))
        Xb, yb = load_batch_dat(p)
        db = np.full((Xb.shape[0],), bn - 1, dtype=np.int64)  # domain: 0..9
        X_list.append(Xb)
        y_list.append(yb)
        d_list.append(db)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    d = np.concatenate(d_list, axis=0)
    return X, y, d


class GasDriftDataset(Dataset):
    def __init__(self, X, y, d):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.d = torch.from_numpy(d).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.d[idx]


# =========================
# 2) Gradient Reversal Layer
# =========================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd):
    return GradReverse.apply(x, lambd)


# =========================
# 3) DANN Model
# =========================
class Encoder(nn.Module):
    def __init__(self, in_dim=128, hidden=256, z_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, x):
        return self.net(x)


class Classifier(nn.Module):
    def __init__(self, z_dim=64, n_classes=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_classes),
        )

    def forward(self, z):
        return self.net(z)


class DomainDiscriminator(nn.Module):
    def __init__(self, z_dim=64, n_domains=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_domains),
        )

    def forward(self, z_grl):
        return self.net(z_grl)


class DANN(nn.Module):
    def __init__(self, in_dim=128, z_dim=64, n_classes=6, n_domains=10):
        super().__init__()
        self.encoder = Encoder(in_dim=in_dim, z_dim=z_dim)
        self.classifier = Classifier(z_dim=z_dim, n_classes=n_classes)
        self.domain_disc = DomainDiscriminator(z_dim=z_dim, n_domains=n_domains)

    def forward(self, x, grl_lambda=1.0):
        z = self.encoder(x)
        y_logits = self.classifier(z)
        z_rev = grad_reverse(z, grl_lambda)
        d_logits = self.domain_disc(z_rev)
        return y_logits, d_logits, z


# =========================
# 4) Train / Eval
# =========================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total, correct_y, correct_d = 0, 0, 0
    for x, y, d in loader:
        x, y, d = x.to(DEVICE), y.to(DEVICE), d.to(DEVICE)
        y_logits, d_logits, _ = model(x, grl_lambda=0.0)
        y_pred = y_logits.argmax(dim=1)
        d_pred = d_logits.argmax(dim=1)
        total += x.size(0)
        correct_y += (y_pred == y).sum().item()
        correct_d += (d_pred == d).sum().item()
    return correct_y / total, correct_d / total


def grl_lambda_schedule(p):
    """
    DANN에서 흔히 쓰는 스케줄:
      lambda = 2/(1+exp(-10p)) - 1
    p: 0 -> 1 (학습 진행률)
    """
    return 2.0 / (1.0 + math.exp(-10 * p)) - 1.0


def train_dann(
    data_dir: str,
    epochs=50,
    batch_size=256,
    lr=1e-3,
    test_batch_holdout=10,  # batch10을 통째로 "미래 데이터"처럼 테스트로 빼고 싶으면 사용
    holdout_entire_batch=True,
):
    X, y, d = load_all_batches(data_dir)

    # -------------------------
    # Split strategy
    # -------------------------
    if holdout_entire_batch:
        # 예: batch10 전체를 test로 (가장 현실적인 "미래 배치" 시나리오)
        test_domain = test_batch_holdout - 1
        test_mask = (d == test_domain)
        X_train, y_train, d_train = X[~test_mask], y[~test_mask], d[~test_mask]
        X_test, y_test, d_test = X[test_mask], y[test_mask], d[test_mask]

        # train -> train/val
        X_tr, X_val, y_tr, y_val, d_tr, d_val = train_test_split(
            X_train, y_train, d_train,
            test_size=0.15, random_state=42, stratify=y_train
        )
    else:
        # 모든 배치에서 섞어서 split (원하면 사용)
        X_tr, X_tmp, y_tr, y_tmp, d_tr, d_tmp = train_test_split(
            X, y, d, test_size=0.30, random_state=42, stratify=y
        )
        X_val, X_test, y_val, y_test, d_val, d_test = train_test_split(
            X_tmp, y_tmp, d_tmp, test_size=0.50, random_state=42, stratify=y_tmp
        )

    # -------------------------
    # Standardize (train 기준)
    # -------------------------
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    train_ds = GasDriftDataset(X_tr, y_tr, d_tr)
    val_ds = GasDriftDataset(X_val, y_val, d_val)
    test_ds = GasDriftDataset(X_test, y_test, d_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # -------------------------
    # Model / Optim
    # -------------------------
    model = DANN(in_dim=128, z_dim=64, n_classes=6, n_domains=10).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    cls_criterion = nn.CrossEntropyLoss()
    dom_criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None

    # -------------------------
    # Training loop
    # -------------------------
    steps_per_epoch = len(train_loader)
    global_step = 0
    total_steps = epochs * steps_per_epoch

    for epoch in range(1, epochs + 1):
        model.train()
        for x, yb, db in train_loader:
            x, yb, db = x.to(DEVICE), yb.to(DEVICE), db.to(DEVICE)

            p = global_step / max(1, total_steps - 1)
            lambd = grl_lambda_schedule(p)

            y_logits, d_logits, _ = model(x, grl_lambda=lambd)

            loss_cls = cls_criterion(y_logits, yb)
            loss_dom = dom_criterion(d_logits, db)

            # 핵심: classifier는 맞추고, domain은 못 맞추게(encoder 기준)
            loss = loss_cls + loss_dom

            opt.zero_grad()
            loss.backward()
            opt.step()

            global_step += 1

        val_cls_acc, val_dom_acc = evaluate(model, val_loader)
        test_cls_acc, test_dom_acc = evaluate(model, test_loader)

        print(
            f"[Epoch {epoch:03d}] "
            f"VAL cls_acc={val_cls_acc:.4f} dom_acc={val_dom_acc:.4f} | "
            f"TEST cls_acc={test_cls_acc:.4f} dom_acc={test_dom_acc:.4f}"
        )

        # val에서 class accuracy 최고 모델 저장
        if val_cls_acc > best_val_acc:
            best_val_acc = val_cls_acc
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
            }

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    final_test_cls_acc, final_test_dom_acc = evaluate(model, test_loader)
    print(f"\n[FINAL] TEST cls_acc={final_test_cls_acc:.4f}, dom_acc={final_test_dom_acc:.4f}")

    return model, scaler


if __name__ == "__main__":
    DATA_DIR = "."

    model, scaler = train_dann(
        data_dir=DATA_DIR,
        epochs=60,
        batch_size=256,
        lr=1e-3,
        test_batch_holdout=10,
        holdout_entire_batch=True,
    )

    torch.save({
        "model_state": model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }, "dann_model.pth")




