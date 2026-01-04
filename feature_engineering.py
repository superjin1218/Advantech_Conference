import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. 데이터 전처리 함수 (원본과 동일) ---
def transform_data_row(row):
    """주어진 raw row를 가스 타입, 농도, 특징들로 변환합니다."""
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
    """데이터 파일을 로드하고 전처리합니다."""
    try:
        raw_batch = pd.read_csv(file_name, sep=" ", header=None)
        transformed_batch = raw_batch.apply(transform_data_row, axis=1, result_type='expand')
        transformed_batch = transformed_batch.dropna(subset=['gas_type'])
        transformed_batch['gas_type'] = transformed_batch['gas_type'].astype(int)
        
        feature_columns = [f"feature_{i}" for i in range(1, 129)]
        
        # DataFrame에 존재하는 특징 열만 선택
        existing_feature_cols = [col for col in feature_columns if col in transformed_batch.columns]
        transformed_batch[existing_feature_cols] = transformed_batch[existing_feature_cols].fillna(0)
        
        # 배치 ID 추가
        transformed_batch['batch'] = batch_id
        
        # 모든 배치가 동일한 열을 갖도록 누락된 특징 열을 0으로 추가
        for col in feature_columns:
            if col not in transformed_batch.columns:
                transformed_batch[col] = 0.0
                
        # 열 순서 고정
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
    """
    16개 센서별로 '반응 모양(EMA 분산)'과 '드리프트 지표' 특징을 생성합니다.
    (f1=DR, f2=DR_norm, f3-f8=EMA) * 16개 센서
    """
    print(f"배치 {df['batch'].iloc[0]}의 특징 공학 수행 중...")
    
    # 0으로 나누기 방지를 위한 작은 값
    epsilon = 1e-6 
    
    for k in range(16): # 16개 센서 루프 (k=0~15)
        sensor_prefix = f'sensor_{k+1}'
        
        # 1. 반응 강도 (DR_norm) 특징 (X축)
        # 센서 1(k=0) -> feature 2
        # 센서 2(k=1) -> feature 10
        # 센서 3(k=2) -> feature 18 ... (k*8 + 2)
        dr_norm_col = f'feature_{k*8 + 2}'
        
        # 2. 반응 모양 (EMA 분산) 특징 (Y축)
        # 센서 1(k=0) -> feature 3~8
        # 센서 2(k=1) -> feature 11~16 ... (k*8 + 3) ~ (k*8 + 8)
        ema_cols = [f'feature_{k*8 + j}' for j in range(3, 9)]
        
        # 6개 EMA 값의 분산(Variance) 계산
        df[f'{sensor_prefix}_ema_var'] = df[ema_cols].var(axis=1)
        
        # 3. 드리프트 지표 (비율) 특징 (보조 지표)
        # 우리가 논의한 (Y축 / X축)
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
    
    # 모든 배치에 대해 새로운 특징 생성
    all_batches_featured = [create_sensor_aware_features(df.copy()) for df in all_batches]
    
    # 훈련(1-6) / 테스트(7-10) 데이터 분리
    train_df = pd.concat(all_batches_featured[:6], ignore_index=True)
    test_dfs_list = all_batches_featured[7:]
    
    # --- 4. 베이스라인 모델 (기존 특징) 훈련 및 평가 ---
    # 원본 128개 특징 + 'batch' 특징
    baseline_features = [f'feature_{i}' for i in range(1, 129)] + ['batch']
    
    X_train_base = train_df[baseline_features]
    y_train = train_df['concentration']
    
    # 스케일링
    scaler_base = StandardScaler()
    X_train_base_scaled = scaler_base.fit_transform(X_train_base)
    
    print("\n[베이스라인 모델 (기존 특징) 훈련 중...]")
    # XGBoost 모델 사용 (원본 노트북의 베스트 모델)
    # (n_estimators=100은 테스트용, 실제로는 튜닝 필요)
    model_base = xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100)
    # model_base = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100) # 대안
    
    model_base.fit(X_train_base_scaled, y_train)
    print("베이스라인 모델 훈련 완료.")

    print("\n--- 베이스라인 모델 (기존 특징) 검증 결과 ---")
    for test_df in test_dfs_list:
        X_test_base = test_df[baseline_features]
        y_test = test_df['concentration']
        X_test_base_scaled = scaler_base.transform(X_test_base)
        
        y_pred_base = model_base.predict(X_test_base_scaled)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
        print(f"  배치 {test_df['batch'].iloc[0]} RMSE: {rmse_base:.2f}")

    # --- 5. 새 모델 (센서 인식 특징) 훈련 및 평가 ---
    # 새로 만든 특징 이름 리스트
    new_ema_features = [f'sensor_{i}_ema_var' for i in range(1, 17)]
    new_ind_features = [f'sensor_{i}_drift_ind' for i in range(1, 17)]
    
    # 기존 특징 + 새 특징 + 'batch' 특징
    advanced_features = baseline_features + new_ema_features + new_ind_features
    
    X_train_adv = train_df[advanced_features]
    # y_train은 동일
    
    # 스케일링
    scaler_adv = StandardScaler()
    X_train_adv_scaled = scaler_adv.fit_transform(X_train_adv)
    
    print("\n[고급 모델 (센서 인식 특징) 훈련 중...]")
    model_adv = xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100)
    # model_adv = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100) # 대안
    
    model_adv.fit(X_train_adv_scaled, y_train)
    print("고급 모델 훈련 완료.")
    
    print("\n--- 고급 모델 (센서 인식 특징) 검증 결과 ---")
    results_summary = []
    for test_df in test_dfs_list:
        X_test_base = test_df[baseline_features]
        y_test = test_df['concentration']
        X_test_base_scaled = scaler_base.transform(X_test_base)
        y_pred_base = model_base.predict(X_test_base_scaled)
        rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))

        X_test_adv = test_df[advanced_features]
        X_test_adv_scaled = scaler_adv.transform(X_test_adv)
        y_pred_adv = model_adv.predict(X_test_adv_scaled)
        rmse_adv = np.sqrt(mean_squared_error(y_test, y_pred_adv))
        
        improvement = (rmse_base - rmse_adv) / rmse_base * 100
        
        print(f"  배치 {test_df['batch'].iloc[0]} RMSE: {rmse_adv:.2f} (개선율: {improvement:.2f}%)")
        results_summary.append({
            "Batch": test_df['batch'].iloc[0],
            "Baseline RMSE": rmse_base,
            "Advanced RMSE": rmse_adv,
            "Improvement (%)": improvement
        })
        
    print("\n--- 최종 요약 ---")
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False, float_format="%.2f"))

# 스크립트 실행
if __name__ == "__main__":
    main()