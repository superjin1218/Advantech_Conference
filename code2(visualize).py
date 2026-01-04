import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. 데이터 전처리 함수 (이전과 동일) ---
def transform_data_row(row):
    try:
        gas_type, concentration = row[0].split(';')
        concentration = float(concentration)
        gas_type = int(gas_type)
        features = {}
        for i in range(1, len(row)):
            if pd.notna(row[i]):
                feature_index, feature_value = row[i].split(':')
                features[f"feature_{int(feature_index)}"] = float(feature_value)
        return {"gas_type": gas_type, "concentration": concentration, **features}
    except Exception as e:
        return None

def load_and_transform(file_name, batch_id):
    try:
        raw_batch = pd.read_csv(file_name, sep=" ", header=None)
        transformed_batch = raw_batch.apply(transform_data_row, axis=1, result_type='expand')
        transformed_batch = transformed_batch.dropna(subset=['gas_type'])
        transformed_batch['gas_type'] = transformed_batch['gas_type'].astype(int)

        feature_columns = [f"feature_{i}" for i in range(1, 129)]
        existing_feature_cols = [col for col in feature_columns if col in transformed_batch.columns]
        transformed_batch[existing_feature_cols] = transformed_batch[existing_feature_cols].fillna(0)
        transformed_batch['batch'] = batch_id

        for col in feature_columns:
            if col not in transformed_batch.columns:
                transformed_batch[col] = 0.0

        final_columns_fixed = ['gas_type', 'concentration', 'batch'] + feature_columns
        return transformed_batch[final_columns_fixed]
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_name}")
        return pd.DataFrame()
    except Exception as e:
        print(f"파일 처리 중 오류 발생 {file_name}: {e}")
        return pd.DataFrame()

# --- 2. '센서 인식' 특징 공학 함수 (이전과 동일) ---
def create_sensor_aware_features(df):
    """16개 센서별로 특징 생성"""
    print(f"배치 {df['batch'].iloc[0]}의 특징 공학 수행 중...")
    epsilon = 1e-6
    for k in range(16):
        sensor_prefix = f'sensor_{k+1}'
        dr_norm_col = f'feature_{k*8 + 2}'
        ema_cols = [f'feature_{k*8 + j}' for j in range(3, 9)]
        df[f'{sensor_prefix}_ema_var'] = df[ema_cols].var(axis=1)
        df[f'{sensor_prefix}_ema_var'] = df[f'{sensor_prefix}_ema_var'].fillna(0)
        df[f'{sensor_prefix}_drift_ind'] = df[f'{sensor_prefix}_ema_var'] / (df[dr_norm_col].fillna(0) + epsilon)

    return df

# --- 3. 메인 시각화 로직 (배치별 t-SNE) ---
def visualize_batches_tsne():
    print("데이터 로드 및 전처리 중...")
    file_names = [f'batch{i}.dat' for i in range(1, 11)]
    all_batches = [load_and_transform(file, i+1) for i, file in enumerate(file_names)] # batch ID 1~10

    if all(df.empty for df in all_batches):
        print("데이터 로드 실패.")
        return

    print("데이터 로드 완료. '센서 인식' 특징 생성 중...")
    all_batches_featured = [create_sensor_aware_features(df.copy()) for df in all_batches]

    # --- 특징 정의 ---
    base_features = [f'feature_{i}' for i in range(1, 129)]
    new_ema_features = [f'sensor_{i}_ema_var' for i in range(1, 17)]
    new_ind_features = [f'sensor_{i}_drift_ind' for i in range(1, 17)]
    # 'batch' 번호는 각 배치 내에서는 상수이므로 t-SNE 계산 시 제외
    advanced_feature_set = base_features + new_ema_features + new_ind_features

    # 가스 종류 이름 및 색상 정의
    class_name = {1: 'Ethanol', 2: 'Ethylene', 3: 'Ammonia', 4: 'Acetaldehyde', 5: 'Acetone', 6: 'Toluene'}
    colors = list(mcolors.TABLEAU_COLORS.keys())
    handles = []
    labels = []

    # --- t-SNE 적용 및 시각화 (배치별) ---
    print("배치별 t-SNE 계산 및 시각화 중...")
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten() # 2x5 배열을 1차원으로 만듦

    for i, batch_df in enumerate(all_batches_featured):
        print(f"Processing Batch {i+1}...")
        X = batch_df[advanced_feature_set].fillna(0)
        y = batch_df['gas_type']

        if X.empty:
            print(f"Batch {i+1} is empty, skipping.")
            axes[i].set_title(f'Batch {i+1} (No Data)')
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            continue

        # 1. 스케일링 (해당 배치 데이터 기준)
        scaler = StandardScaler()
        # 데이터가 1개인 경우 스케일링 불가 -> 원본 사용
        if len(X) > 1:
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = X.values # DataFrame을 NumPy 배열로 변환

        # 2. t-SNE 적용
        # 데이터 수가 perplexity보다 작거나 같으면 에러 발생 -> perplexity 조정
        perplexity_value = min(30, len(X_scaled) - 1) if len(X_scaled) > 1 else 5
        if perplexity_value < 5: # perplexity는 최소 5 이상이어야 함
              perplexity_value = 5

        # 데이터 수가 너무 적으면(e.g., 1개) t-SNE 실행 불가
        if len(X_scaled) > 3 : # 보통 t-SNE는 샘플 수가 어느 정도 있어야 의미 있음
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, n_iter=300, init='pca')
                X_tsne = tsne.fit_transform(X_scaled)
            except ValueError as e:
                print(f" Batch {i+1} t-SNE Error: {e}. Skipping t-SNE.")
                X_tsne = np.zeros((len(X_scaled), 2)) # 오류 시 (0,0)에 점 찍기
        else:
             print(f" Batch {i+1} has too few samples ({len(X_scaled)}) for t-SNE. Skipping.")
             X_tsne = np.zeros((len(X_scaled), 2)) # 샘플 적을 시 (0,0)에 점 찍기

        # 3. 시각화
        ax = axes[i]
        unique_gas_types = sorted(y.unique())

        for gas_label in unique_gas_types:
            indices = (y == gas_label)
            class_label_name = class_name.get(gas_label, f'Unknown {gas_label}')
            color_index = gas_label % len(colors)
            scatter = ax.scatter(X_tsne[indices, 0], X_tsne[indices, 1],
                                   color=colors[color_index], alpha=0.7, label=class_label_name)

            # 첫 번째 그래프(i=0)에서만 범례 정보 수집
            if i == 0 and class_label_name not in labels:
                 handles.append(scatter)
                 labels.append(class_label_name)

        ax.set_title(f't-SNE for Batch {i+1}')
        ax.set_xlabel('t-SNE Component 1')
        ax.set_ylabel('t-SNE Component 2')
        ax.grid(True)

    # 전체 범례 추가
    fig.legend(handles=handles, labels=labels, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, -0.05))
    plt.suptitle('Batch-wise t-SNE Visualization using Advanced Features', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 범례 공간 확보
    plt.show()

# 스크립트 실행
if __name__ == "__main__":
    visualize_batches_tsne()