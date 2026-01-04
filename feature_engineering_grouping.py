import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb # RandomForest 대신 사용 가능
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. 데이터 전처리 함수 (원본과 동일) ---
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

# --- 2. '센서 인식' 특징 공학 함수 (핵심) ---
def create_sensor_aware_features(df):
    """16개 센서별로 '반응 모양(EMA 분산)'과 '드리프트 지표' 특징을 생성합니다."""
    print(f"배치 {df['batch'].iloc[0]}의 특징 공학 수행 중...")
    epsilon = 1e-6 
    
    for k in range(16): # 16개 센서
        sensor_prefix = f'sensor_{k+1}'
        
        # 1. 반응 강도 (DR_norm) (X축)
        dr_norm_col = f'feature_{k*8 + 2}'
        
        # 2. 반응 모양 (EMA 분산) (Y축)
        ema_cols = [f'feature_{k*8 + j}' for j in range(3, 9)]
        
        # EMA 값의 분산(Variance) 계산 (빠른 반응=분산 높음, 느린 반응=분산 낮음)
        df[f'{sensor_prefix}_ema_var'] = df[ema_cols].var(axis=1)
        
        # 3. 드리프트 지표 (비율) (보조 지표)
        df[f'{sensor_prefix}_drift_ind'] = df[f'{sensor_prefix}_ema_var'] / (df[dr_norm_col] + epsilon)
        
    return df

# --- 3. 메인 실행 로직 ---
def main():
    print("데이터 로드 및 전처리 중...")
    file_names = [f'batch{i}.dat' for i in range(1, 11)]
    all_batches = [load_and_transform(file, i+1) for i, file in enumerate(file_names)]
    
    if all(df.empty for df in all_batches):
        print("데이터 로드 실패. 스크립트를 종료합니다.")
        return
    
    print("데이터 로드 완료. '센서 인식' 특징 생성 중...")
    
    # 모든 배치(1~10)에 대해 새로운 특징 생성
    all_batches_featured = [create_sensor_aware_features(df.copy()) for df in all_batches]
    
    # 모든 데이터를 하나의 DataFrame으로 결합
    full_data = pd.concat(all_batches_featured, ignore_index=True)

    # --- 4. 그룹별 모델 훈련 및 검증 ---
    
    # 새로 만든 특징 이름 리스트
    base_features = [f'feature_{i}' for i in range(1, 129)]
    new_ema_features = [f'sensor_{i}_ema_var' for i in range(1, 17)]
    new_ind_features = [f'sensor_{i}_drift_ind' for i in range(1, 17)]
    
    # 최종 특징 리스트: 기존 128 + 새 특징 32 + 'batch'
    # 이것이 드리프트를 해석하는 모델의 입력값
    advanced_features = base_features + new_ema_features + new_ind_features + ['batch']

    # 그룹 정의
    group_defs = {
        "G1(에탄올)": (full_data['gas_type'] == 1),
        "G2(아세트/톨루엔)": (full_data['gas_type'].isin([4, 6])),
        "G3(기타)": (full_data['gas_type'].isin([2, 3, 5]))
    }
    
    # 훈련(1-7) / 검증(8-10) 분리 마스크
    train_mask = (full_data['batch'] <= 7)
    validation_mask = (full_data['batch'] > 7)
    
    # 결과를 저장할 리스트
    results_summary = []
    
    # 이전 실행(베이스라인)의 RMSE 값 (비교용)
    baseline_rmse = {
        8: {"G1(에탄올)": 68.32, "G2(아세트/톨루엔)": 68.32, "G3(기타)": 68.32}, # 이전엔 통합이었으므로 G1, G2, G3 모두 같은 값으로 가정
        9: {"G1(에탄올)": 36.11, "G2(아세트/톨루엔)": 36.11, "G3(기타)": 36.11},
        10: {"G1(에탄올)": 218.50, "G2(아세트/톨루엔)": 218.50, "G3(기타)": 218.50}
    }
    # 참고: 이전 코드는 G1,G2,G3를 나누지 않아서 68.32, 36.11, 218.50 이라는 *전체 평균* RMSE가 나왔습니다.
    # 이번엔 그룹별로 따로 RMSE를 계산하므로 더 정확한 비교가 됩니다.
    # (이해를 돕기 위해 이전 통합 RMSE를 그대로 사용합니다)


    print("\n--- 최종 전략 (그룹 분리 + 특징 공학) 훈련 및 검증 ---")

    for group_name, group_mask in group_defs.items():
        print(f"\n[{group_name} 모델 훈련 중...]")
        
        # 1. 훈련/검증 데이터 분리
        X_train = full_data[train_mask & group_mask][advanced_features]
        y_train = full_data[train_mask & group_mask]['concentration']
        
        X_valid = full_data[validation_mask & group_mask][advanced_features]
        y_valid = full_data[validation_mask & group_mask]['concentration']
        
        if X_train.empty or X_valid.empty:
            print(f"{group_name}: 훈련 또는 검증 데이터가 없습니다. 건너뜁니다.")
            continue
            
        # 2. 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        # 3. 모델 훈련 (RandomForest 사용)
        model = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100)
        # model = xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100) # XGBoost 사용 시
        
        model.fit(X_train_scaled, y_train)
        print(f"[{group_name} 모델 훈련 완료.]")
        
        # 4. 검증 (배치 8, 9, 10별로 따로따로)
        print(f"[{group_name} 모델 검증 중...]")
        for batch_num in [8, 9, 10]:
            batch_mask = (X_valid['batch'] == batch_num)
            
            X_valid_batch = X_valid[batch_mask]
            y_valid_batch = y_valid[batch_mask]
            
            if X_valid_batch.empty:
                continue
                
            X_valid_batch_scaled = scaler.transform(X_valid_batch)
            y_pred = model.predict(X_valid_batch_scaled)
            
            rmse_adv = np.sqrt(mean_squared_error(y_valid_batch, y_pred))
            
            # 비교를 위해 이전 통합 RMSE 값을 가져옴
            rmse_base = baseline_rmse[batch_num][group_name]    
            improvement = (rmse_base - rmse_adv) / rmse_base * 100
            
            results_summary.append({
                "Batch": batch_num,
                "Group": group_name,
                "Baseline RMSE (전체)": rmse_base,
                "Advanced RMSE (그룹)": rmse_adv,
                "Improvement (%)": improvement
            })

    # --- 5. 최종 결과 요약 ---
    print("\n" + "="*50)
    print("--- 최종 요약 (그룹별 전문가 모델) ---")
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.sort_values(by=['Batch', 'Group']).to_string(index=False, float_format="%.2f"))
        print("\n개선율(%)이 높을수록 드리프트 해석에 성공한 것입니다.")
    else:
        print("결과 요약에 표시할 데이터가 없습니다.")

# 스크립트 실행
if __name__ == "__main__":
    main()