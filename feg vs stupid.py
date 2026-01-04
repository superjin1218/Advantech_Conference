import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb # RandomForest 대신 사용 가능
import warnings

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# --- 1. 데이터 전처리 함수 (동일) ---
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

# --- 2. '센서 인식' 특징 공학 함수 (동일) ---
def create_sensor_aware_features(df):
    """16개 센서별로 '반응 모양(EMA 분산)'과 '드리프트 지표' 특징을 생성합니다."""
    print(f"배치 {df['batch'].iloc[0]}의 특징 공학 수행 중...")
    epsilon = 1e-6 
    
    for k in range(16): # 16개 센서
        sensor_prefix = f'sensor_{k+1}'
        dr_norm_col = f'feature_{k*8 + 2}'
        ema_cols = [f'feature_{k*8 + j}' for j in range(3, 9)]
        df[f'{sensor_prefix}_ema_var'] = df[ema_cols].var(axis=1)
        df[f'{sensor_prefix}_ema_var'] = df[f'{sensor_prefix}_ema_var'].fillna(0) # 분산 NaN 0으로 처리
        df[f'{sensor_prefix}_drift_ind'] = df[f'{sensor_prefix}_ema_var'] / (df[dr_norm_col].fillna(0) + epsilon)
        
    return df

# --- 3. 메인 실행 로직 (수정됨) ---
def main():
    print("데이터 로드 및 전처리 중...")
    file_names = [f'batch{i}.dat' for i in range(1, 11)]
    all_batches = [load_and_transform(file, i+1) for i, file in enumerate(file_names)]
    
    if all(df.empty for df in all_batches):
        print("데이터 로드 실패. 스크립트를 종료합니다.")
        return
    
    print("데이터 로드 완료. '센서 인식' 특징 생성 중...")
    all_batches_featured = [create_sensor_aware_features(df.copy()) for df in all_batches]
    full_data = pd.concat(all_batches_featured, ignore_index=True)

    # --- 4. 특징 및 그룹 정의 ---
    
    # "멍청한" 모델이 사용할 특징 (특징 공학 X)
    base_features = [f'feature_{i}' for i in range(1, 129)] + ['batch']
    
    # "똑똑한" 모델이 사용할 특징 (특징 공학 O)
    new_ema_features = [f'sensor_{i}_ema_var' for i in range(1, 17)]
    new_ind_features = [f'sensor_{i}_drift_ind' for i in range(1, 17)]
    advanced_features = base_features + new_ema_features + new_ind_features

    group_defs = {
        "G1(에탄올)": (full_data['gas_type'] == 1),
        "G2(아세트/톨루엔)": (full_data['gas_type'].isin([4, 6])),
        "G3(기타)": (full_data['gas_type'].isin([2, 3, 5]))
    }
        
    train_mask = (full_data['batch'] <= 7)
    validation_mask = (full_data['batch'] > 7)
    
    results_summary = []
    
    # 딕셔너리를 생성하여 '진짜 베이스라인' RMSE 값을 저장
    baseline_rmse_scores = {}

    print("\n--- 1. '진짜 베이스라인' (멍청한 전문가) 모델 훈련 및 평가 ---")
    
    for group_name, group_mask in group_defs.items():
        print(f"\n[{group_name} '멍청한' 모델 훈련 중... (특징 공학 X)]")
        
        X_train = full_data[train_mask & group_mask][base_features] # 'base_features' 사용
        y_train = full_data[train_mask & group_mask]['concentration']
        
        X_valid = full_data[validation_mask & group_mask][base_features] # 'base_features' 사용
        
        if X_train.empty or X_valid.empty:
            continue
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        model_base = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100)
        model_base.fit(X_train_scaled, y_train)
        
        print(f"[{group_name} '멍청한' 모델 검증 중...]")
        for batch_num in [8, 9, 10]:
            batch_mask = (X_valid['batch'] == batch_num)
            y_valid_batch = full_data[validation_mask & group_mask & (full_data['batch'] == batch_num)]['concentration']
            
            if batch_mask.sum() == 0:
                continue
                
            X_valid_batch_scaled = scaler.transform(X_valid[batch_mask])
            y_pred_base = model_base.predict(X_valid_batch_scaled)
            rmse_base = np.sqrt(mean_squared_error(y_valid_batch, y_pred_base))
            
            # (배치, 그룹)을 키로 사용하여 RMSE 값 저장
            baseline_rmse_scores[(batch_num, group_name)] = rmse_base
            print(f"  배치 {batch_num} RMSE: {rmse_base:.2f}")


    print("\n--- 2. '어드밴스드' (똑똑한 전문가) 모델 훈련 및 평가 ---")

    for group_name, group_mask in group_defs.items():
        print(f"\n[{group_name} '똑똑한' 모델 훈련 중... (특징 공학 O)]")
        
        X_train = full_data[train_mask & group_mask][advanced_features] # 'advanced_features' 사용
        y_train = full_data[train_mask & group_mask]['concentration']
        
        X_valid = full_data[validation_mask & group_mask][advanced_features] # 'advanced_features' 사용
        
        if X_train.empty or X_valid.empty:
            continue
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        
        model_adv = RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100)
        model_adv.fit(X_train_scaled, y_train)
        
        print(f"[{group_name} '똑똑한' 모델 검증 중...]")
        for batch_num in [8, 9, 10]:
            batch_mask = (X_valid['batch'] == batch_num)
            y_valid_batch = full_data[validation_mask & group_mask & (full_data['batch'] == batch_num)]['concentration']
            
            if batch_mask.sum() == 0:
                continue
                
            X_valid_batch_scaled = scaler.transform(X_valid[batch_mask])
            y_pred_adv = model_adv.predict(X_valid_batch_scaled)
            
            rmse_adv = np.sqrt(mean_squared_error(y_valid_batch, y_pred_adv))
            
            # 저장해둔 '진짜 베이스라인' RMSE 값을 불러옴
            rmse_base = baseline_rmse_scores.get((batch_num, group_name), 0)
            improvement = (rmse_base - rmse_adv) / rmse_base * 100 if rmse_base > 0 else 0
            
            results_summary.append({
                "Batch": batch_num,
                "Group": group_name,
                "Baseline RMSE (멍청한 전문가)": rmse_base,
                "Advanced RMSE (똑똑한 전문가)": rmse_adv,
                "Improvement (%)": improvement
            })

    # --- 5. 최종 결과 요약 ---
    print("\n" + "="*50)
    print("--- 최종 요약 (진짜 Baseline vs 똑똑한 전문가) ---")
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        print(summary_df.sort_values(by=['Batch', 'Group']).to_string(index=False, float_format="%.2f"))
        print("\n'특징 공학'의 힘으로 얼마나 개선되었는지(%) 보여줍니다.")
    else:
        print("결과 요약에 표시할 데이터가 없습니다.")

# 스크립트 실행
if __name__ == "__main__":
    main()