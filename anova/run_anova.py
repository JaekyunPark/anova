import pandas as pd
import numpy as np
import scipy.stats as stats
import pyreadstat
import sys
import os

def load_spss_file(file_path):
    """
    SPSS 파일을 불러옵니다.
    """
    try:
        df, meta = pyreadstat.read_sav(file_path)
        return df, meta
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None, None

def calculate_gg_epsilon(Y, weights):
    n, k = Y.shape
    if k <= 1: return 1.0
    k_range = np.arange(1, k + 1)
    helmert = np.zeros((k, k - 1))
    for i in range(k - 1):
        helmert[:i+1, i] = 1.0
        helmert[i+1, i] = -(i + 1)
        helmert[:, i] /= np.sqrt((i + 1)**2 + (i + 1))
    contrasts = Y @ helmert
    sum_w = np.sum(weights)
    mean_contrasts = np.average(contrasts, axis=0, weights=weights)
    diff = contrasts - mean_contrasts
    S = (diff.T * weights) @ diff / sum_w
    tr_S, tr_S2 = np.trace(S), np.trace(S @ S)
    if tr_S2 == 0: return 1.0
    epsilon = (tr_S**2) / ((k - 1) * tr_S2)
    return min(1.0, max(1.0 / (k - 1), epsilon))

def calculate_hf_epsilon(gg_eps, n, k):
    num = n * (k - 1) * gg_eps - 2
    den = (k - 1) * (n - 1 - (k - 1) * gg_eps)
    if den <= 0: return 1.0
    epsilon = num / den
    return min(1.0, max(gg_eps, epsilon))

def weighted_repeated_measures_anova(df, dep_vars, weight_col, normalize=True, use_weighted_df=False):
    data = df[dep_vars + [weight_col]].dropna()
    if len(data) == 0: return {'F': np.nan, 'p_gg': np.nan, 'eps_gg': 1.0, 'eps_hf': 1.0, 'weighted_n': 0}
    n_samples, n_conditions = len(data), len(dep_vars)
    weights = data[weight_col].values
    weighted_n = np.sum(weights)
    if normalize:
        current_weights = weights * n_samples / weighted_n
        current_sum_w = n_samples
    else:
        current_weights = weights
        current_sum_w = weighted_n
    Y = data[dep_vars].values
    mean_j = np.average(Y, axis=0, weights=current_weights)
    grand_mean = np.average(Y, weights=np.tile(current_weights[:, None], (1, n_conditions)))
    ss_total = np.sum(current_weights[:, None] * (Y - grand_mean)**2)
    subject_means = np.mean(Y, axis=1)
    ss_subjects = np.sum(current_weights * n_conditions * (subject_means - grand_mean)**2)
    ss_time = current_sum_w * np.sum((mean_j - grand_mean)**2)
    ss_error = ss_total - ss_subjects - ss_time
    df_time = n_conditions - 1
    effective_n = weighted_n if use_weighted_df else n_samples
    df_error = (effective_n - 1) * (n_conditions - 1)
    ms_time, ms_error = ss_time / df_time, ss_error / df_error
    f_value = ms_time / ms_error if ms_error != 0 else 0
    eps_gg = calculate_gg_epsilon(Y, current_weights)
    p_gg = stats.f.sf(f_value, df_time * eps_gg, df_error * eps_gg) if df_error > 0 else np.nan
    eps_hf = calculate_hf_epsilon(eps_gg, effective_n, n_conditions)
    p_hf = stats.f.sf(f_value, df_time * eps_hf, df_error * eps_hf) if df_error > 0 else np.nan
    return {
        'F': f_value, 'p_gg': p_gg, 'p_hf': p_hf, 'eps_gg': eps_gg, 'eps_hf': eps_hf,
        'df_model': df_time, 'df_error': df_error, 'weighted_n': weighted_n
    }

def run_analysis():
    # 설정 (사용자가 코드 내에서 수정하거나 인자로 받을 수 있도록 구성)
    # 실제 사용 시에는 이 부분을 파일 경로에 맞게 수정해야 합니다.
    print("=== SPSS Weighted Repeated Measures ANOVA Program ===")
    
    # 1. 파일 불러오기
    # 현재 디렉토리의 .sav 파일을 찾습니다.
    files = [f for f in os.listdir('.') if f.endswith('.sav')]
    if not files:
        print("현재 디렉토리에 .sav 파일이 없습니다.")
        # 테스트를 위해 임의의 데이터 생성 로직을 넣을 수도 있습니다.
        file_path = input("SPSS 파일 경로를 입력하세요 (.sav): ").strip()
    else:
        print(f"발견된 파일: {files[0]}")
        file_path = files[0] # 첫 번째 파일 사용
        
    if not os.path.exists(file_path):
        print("파일을 찾을 수 없습니다.")
        return

    df, meta = load_spss_file(file_path)
    if df is None:
        return

    print(f"데이터 로드 완료: {len(df)} cases")
    print("변수 목록:", list(df.columns))
    
    try:
        # 사용자 입력 (또는 하드코딩된 설정)
        # 예시를 위해 입력을 받습니다.
        print("\n--- 변수 설정 ---")
        weight_var = input("가중치 변수명을 입력하세요: ").strip()
        dep_vars_input = input("종속변수 3개를 쉼표로 구분하여 입력하세요 (예: var1, var2, var3): ").strip()
        dep_vars = [v.strip() for v in dep_vars_input.split(',')]
        
        banner_vars_input = input("배너 변수(집단 변수)를 쉼표로 구분하여 입력하세요 (선택, 없으면 엔터): ").strip()
        banner_vars = [b.strip() for b in banner_vars_input.split(',')] if banner_vars_input else []
        
        print("\n--- 분석 옵션 ---")
        freq_input = input("SPSS 방식(데이터 물리적 복제)을 사용하시겠습니까? (y/N): ").strip().lower()
        use_frequency_weight = freq_input == 'y'
        
        normalize, use_weighted_df = True, False
        if not use_frequency_weight:
            norm_input = input("가중치 정규화(표본 크기 기준)를 수행하시겠습니까? (Y/n): ").strip().lower()
            normalize = norm_input != 'n'
            df_input = input("P-값 계산 시 가중치 합(모집단 크기)을 자유도로 사용하시겠습니까? (y/N): ").strip().lower()
            use_weighted_df = df_input == 'y'
        
        print("\n" + "="*120)
        print(f"{'Group':<25} | {'W-N':<8} | {'F-Value':<10} | {'Eps(GG)':<8} | {'P(GG)':<10} | {'Eps(HF)':<8} | {'P(HF)':<10} | {'Sig':<5}")
        print("="*120)
        
        # 1. 전체 샘플 분석
        res = weighted_repeated_measures_anova(df, dep_vars, weight_var, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
        sig = "*" if res['p_gg'] < 0.05 else ""
        print(f"{'Total Sample':<25} | {res['weighted_n']:<8.2f} | {res['F']:<10.4f} | {res['eps_gg']:<8.4f} | {res['p_gg']:<10.4f} | {res['eps_hf']:<8.4f} | {res['p_hf']:<10.4f} | {sig:<5}")
        
        # 2. 배너 변수별 분석
        for banner in banner_vars:
            if banner not in df.columns:
                print(f"Warning: Banner variable '{banner}' not found.")
                continue
                
            print("-" * 120)
            print(f"Banner: {banner}")
            
            # 배너 변수의 값 설명(Labels)이 있으면 가져오기
            val_labels = meta.variable_value_labels.get(banner, {}) if meta else {}
            
            valid_df = df[df[banner].notna()]
            groups = sorted(valid_df[banner].unique())
            
            for group_val in groups:
                sub_df = df[df[banner] == group_val]
                
                # 라벨링
                group_label = val_labels.get(group_val, str(group_val))
                display_name = f"  {banner}={group_label}"
                
                res = weighted_repeated_measures_anova(sub_df, dep_vars, weight_var, normalize=normalize, use_weighted_df=use_weighted_df, use_frequency_weight=use_frequency_weight)
                sig = "*" if res['p_gg'] < 0.05 else ""
                
                print(f"{display_name:<25} | {res['weighted_n']:<8.2f} | {res['F']:<10.4f} | {res['eps_gg']:<8.4f} | {res['p_gg']:<10.4f} | {res['eps_hf']:<8.4f} | {res['p_hf']:<10.4f} | {sig:<5}")
                
        print("="*80)
        print("* p < .05")
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_analysis()
