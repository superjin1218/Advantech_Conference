import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

# train_dann.py에서 정의한 것과 동일해야 함
from train_dann import (
    DANN,
    load_all_batches,
    GasDriftDataset,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "."   # batch*.dat 있는 폴더


# ==============================
# Gas label → name / color 매핑
# ==============================
GAS_NAMES = {
    0: "Ethanol",
    1: "Ethylene",
    2: "Ammonia",
    3: "Acetaldehyde",
    4: "Acetone",
    5: "Toluene",
}

GAS_COLORS = {
    0: "orange",        # Ethanol
    1: "green",         # Ethylene
    2: "red",           # Ammonia
    3: "purple",        # Acetaldehyde
    4: "saddlebrown",   # Acetone
    5: "hotpink",       # Toluene
}


# -----------------------------
# 1) 모델 + 스케일러 로드
# -----------------------------
checkpoint = torch.load(
    "dann_model.pth",
    map_location=DEVICE,
    weights_only=False
)

model = DANN(in_dim=128, z_dim=64, n_classes=6, n_domains=10).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

scaler = StandardScaler()
scaler.mean_ = checkpoint["scaler_mean"]
scaler.scale_ = checkpoint["scaler_scale"]


# -----------------------------
# 2) 데이터 로드 (전체 batch)
# -----------------------------
X, y, d = load_all_batches(DATA_DIR)
X = scaler.transform(X)

dataset = GasDriftDataset(X, y, d)
loader = DataLoader(dataset, batch_size=512, shuffle=False)


# -----------------------------
# 3) latent z 추출
# -----------------------------
Z, Y, D = [], [], []
with torch.no_grad():
    for x, yb, db in loader:
        x = x.to(DEVICE)
        _, _, z = model(x, grl_lambda=0.0)
        Z.append(z.cpu().numpy())
        Y.append(yb.numpy())
        D.append(db.numpy())

Z = np.concatenate(Z)
Y = np.concatenate(Y)
D = np.concatenate(D)


# -----------------------------
# 4) 전체 latent t-SNE (Gas 기준, 색 고정)
# -----------------------------
def plot_tsne_gas_fixed(Z, Y, title):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        init="pca",
        random_state=42,
    )
    Z_2d = tsne.fit_transform(Z)

    plt.figure(figsize=(7, 6))
    for gas_id in range(6):
        mask = (Y == gas_id)
        plt.scatter(
            Z_2d[mask, 0],
            Z_2d[mask, 1],
            s=8,
            alpha=0.8,
            color=GAS_COLORS[gas_id],
            label=GAS_NAMES[gas_id],
        )

    plt.legend(title="Gas", markerscale=2)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.show()


# # -----------------------------
# # 5) Batch별 latent t-SNE (색 고정)
# # -----------------------------
# def plot_tsne_per_batch_fixed(Z, Y, D, batch_id, max_points=3000):
#     mask = (D == batch_id)
#     Zb = Z[mask]
#     Yb = Y[mask]

#     if Zb.shape[0] > max_points:
#         idx = np.random.choice(Zb.shape[0], max_points, replace=False)
#         Zb = Zb[idx]
#         Yb = Yb[idx]

#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         learning_rate=200,
#         n_iter=1000,
#         init="pca",
#         random_state=42,
#     )
#     Z_2d = tsne.fit_transform(Zb)

#     plt.figure(figsize=(6, 5))
#     for gas_id in range(6):
#         mask_g = (Yb == gas_id)
#         plt.scatter(
#             Z_2d[mask_g, 0],
#             Z_2d[mask_g, 1],
#             s=10,
#             alpha=0.85,
#             color=GAS_COLORS[gas_id],
#             label=GAS_NAMES[gas_id],
#         )

#     plt.legend(title="Gas", markerscale=1.5)
#     plt.title(f"Latent t-SNE — Batch {batch_id + 1}")
#     plt.xlabel("t-SNE dim 1")
#     plt.ylabel("t-SNE dim 2")
#     plt.tight_layout()
#     plt.show()


# # ============================
# # 실행
# # ============================
# plot_tsne_gas_fixed(Z, Y, "Latent t-SNE (colored by Gas)")

# for b in range(10):
#     plot_tsne_per_batch_fixed(Z, Y, D, batch_id=b)


def plot_tsne_all_batches_grid(Z, Y, D, max_points=2000):
    """
    Batch 1~10을 2x5 subplot 한 화면에 시각화
    색상은 gas 기준으로 고정
    """

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for batch_id in range(10):
        ax = axes[batch_id]

        mask = (D == batch_id)
        Zb = Z[mask]
        Yb = Y[mask]

        if Zb.shape[0] > max_points:
            idx = np.random.choice(Zb.shape[0], max_points, replace=False)
            Zb = Zb[idx]
            Yb = Yb[idx]

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate=200,
            n_iter=1000,
            init="pca",
            random_state=42,
        )
        Z_2d = tsne.fit_transform(Zb)

        for gas_id in range(6):
            mask_g = (Yb == gas_id)
            ax.scatter(
                Z_2d[mask_g, 0],
                Z_2d[mask_g, 1],
                s=8,
                alpha=0.85,
                color=GAS_COLORS[gas_id],
                label=GAS_NAMES[gas_id] if batch_id == 0 else None,
            )

        ax.set_title(f"Batch {batch_id + 1}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # 범례는 한 번만 (맨 위 batch 기준)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=6,
        title="Gas",
        fontsize=11,
        title_fontsize=12,
    )

    fig.suptitle(
        "Latent t-SNE per Batch (Gas-colored, DANN)",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.show()


plot_tsne_all_batches_grid(Z, Y, D)
