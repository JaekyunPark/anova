import pandas as pd
import numpy as np
import scipy.stats as stats
import pyreadstat

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
    """
    Greenhouse-Geisser Epsilon(ε)을 계산합니다.
    Y: (n_samples, n_conditions) 데이터 행렬
    """
    n, k = Y.shape
    if k <= 1:
        return 1.0

    k_range = np.arange(1, k + 1)
    helmert = np.zeros((k, k - 1))
    for i in range(k - 1):
        helmert[:i+1, i] = 1.0
        helmert[i+1, i] = -(i + 1)
        helmert[:, i] /= np.sqrt((i + 1)**2 + (i + 1))

    contrasts = Y @ helmert
    sum_w = np.sum(weights)
    if sum_w == 0:
        return 1.0
        
    mean_contrasts = np.average(contrasts, axis=0, weights=weights)
    diff = contrasts - mean_contrasts
    S = (diff.T * weights) @ diff / sum_w

    tr_S = np.trace(S)
    tr_S2 = np.trace(S @ S)
    
    if tr_S2 == 0:
        return 1.0
        
    epsilon = (tr_S**2) / ((k - 1) * tr_S2)
    return min(1.0, max(1.0 / (k - 1), epsilon))

def calculate_hf_epsilon(gg_epsilon, n, k):
    """
    Huynh-Feldt Epsilon(ε)을 계산합니다.
    """
    num = n * (k - 1) * gg_epsilon - 2
    den = (k - 1) * (n - 1 - (k - 1) * gg_epsilon)
    if den <= 0:
        return 1.0
    epsilon = num / den
    return min(1.0, max(gg_epsilon, epsilon))

def calculate_mauchly_test(Y, weights):
    """
    Mauchly's Test for Sphericity (W, Chi2, p-value)
    """
    n, k = Y.shape
    if k <= 2:
        return 1.0, 0.0, 1.0 # 2년 이하는 구형성 가정이 항상 충족됨

    # 직교 대비 생성
    helmert = np.zeros((k, k - 1))
    for i in range(k - 1):
        helmert[:i+1, i] = 1.0
        helmert[i+1, i] = -(i + 1)
        helmert[:, i] /= np.sqrt((i + 1)**2 + (i + 1))

    contrasts = Y @ helmert
    sum_w = np.sum(weights)
    if sum_w <= k: return 1.0, 0.0, 1.0
    
    mean_contrasts = np.average(contrasts, axis=0, weights=weights)
    diff = contrasts - mean_contrasts
    S = (diff.T * weights) @ diff / sum_w
    
    # Mauchly's W = det(S) / (trace(S)/(k-1))^(k-1)
    det_S = np.linalg.det(S)
    tr_S = np.trace(S)
    
    if tr_S == 0: return 1.0, 0.0, 1.0
    
    mauchly_w = det_S / ((tr_S / (k - 1))**(k - 1))
    mauchly_w = max(1e-10, min(1.0, mauchly_w))
    
    # Chi-square transformation
    # d = 1 - (2(k-1)^2 + (k-1) + 2) / (6(k-1)(n-1))
    d = 1 - (2 * (k - 1)**2 + (k - 1) + 2) / (6 * (k - 1) * (sum_w - 1))
    chi2 = -(sum_w - 1) * d * np.log(mauchly_w)
    df_m = k * (k - 1) / 2 - 1
    
    p_mauchly = stats.chi2.sf(chi2, df_m)
    
    return mauchly_w, chi2, p_mauchly

def weighted_repeated_measures_anova(df, dep_vars, weight_col, normalize=True, use_weighted_df=False, use_frequency_weight=False):
    """
    가중치를 적용한 반복측정 ANOVA 및 구형성 요약 리포트.
    """
    data = df[dep_vars + [weight_col]].dropna()
    
    if len(data) == 0:
        return {
            'F': np.nan, 'p_unc': np.nan, 'p_gg': np.nan, 'p_hf': np.nan, 
            'eps_gg': np.nan, 'eps_hf': np.nan, 'm_w': np.nan, 'm_p': np.nan,
            'df_model': 0, 'df_error': 0, 'Means': [], 'weighted_n': 0
        }

    # 1. SPSS 방식: 물리적 데이터 복제 (Frequency Weighting)
    if use_frequency_weight:
        weights = data[weight_col].round().astype(int)
        valid_idx = weights > 0
        data = data[valid_idx]
        weights = weights[valid_idx]
        
        if len(data) == 0:
            return {
                'F': np.nan, 'p_unc': np.nan, 'p_gg': np.nan, 'p_hf': np.nan, 
                'eps_gg': np.nan, 'eps_hf': np.nan, 'm_w': np.nan, 'm_p': np.nan,
                'df_model': 0, 'df_error': 0, 'Means': [], 'weighted_n': 0
            }

        data = data.loc[data.index.repeat(weights)].reset_index(drop=True)
        n_samples = len(data)
        n_conditions = len(dep_vars)
        Y = data[dep_vars].values
        current_weights = np.ones(n_samples)
        current_sum_w = n_samples
    else:
        # 기존 방식: 임포턴스 가중치 (Reliability Weighting)
        n_samples = len(data)
        n_conditions = len(dep_vars)
        weights = data[weight_col].values
        weighted_n = np.sum(weights)

        if weighted_n == 0:
            return {
                'F': np.nan, 'p_unc': np.nan, 'p_gg': np.nan, 'p_hf': np.nan, 
                'eps_gg': np.nan, 'eps_hf': np.nan, 'm_w': np.nan, 'm_p': np.nan,
                'df_model': 0, 'df_error': 0, 'Means': [], 'weighted_n': 0
            }

        if normalize:
            current_weights = weights * n_samples / weighted_n
            current_sum_w = n_samples
        else:
            current_weights = weights
            current_sum_w = weighted_n
        
        Y = data[dep_vars].values

    # 2. ANOVA 계산
    mean_j = np.average(Y, axis=0, weights=current_weights)
    grand_mean = np.average(Y, weights=np.tile(current_weights[:, None], (1, n_conditions)))

    # 2. 제곱합(Sum of Squares) 계산
    # SS_Total
    ss_total = np.sum(current_weights[:, None] * (Y - grand_mean)**2)
    
    # SS_Subjects (피험자 간 변동)
    subject_means = np.mean(Y, axis=1)
    ss_subjects = np.sum(current_weights * n_conditions * (subject_means - grand_mean)**2)
    
    # SS_Time (시점 간 변동 / 효과)
    ss_time = current_sum_w * np.sum((mean_j - grand_mean)**2)
    
    # SS_Error (오차 / 상호작용)
    ss_error = ss_total - ss_subjects - ss_time
    
    # 3. 자유도 및 평균제곱 계산
    df_time = n_conditions - 1
    
    # SPSS 방식(Frequency)일 때는 이미 복제된 n_samples가 sum(weights)와 같음
    effective_n = current_sum_w if use_frequency_weight else (weighted_n if use_weighted_df else n_samples)
        
    df_error = (effective_n - 1) * (n_conditions - 1)
    
    ms_time = ss_time / df_time
    ms_error = ss_error / df_error
    
    # 4. F-값 및 P-값 (Uncorrected)
    f_value = ms_time / ms_error if ms_error != 0 else 0
    p_unc = stats.f.sf(f_value, df_time, df_error) if df_error > 0 else np.nan

    # 3. 구형성 검정 (Mauchly's Test)
    m_w, m_chi2, m_p = calculate_mauchly_test(Y, current_weights)

    # 4. Greenhouse-Geisser 보정
    # 가중치를 반영한 Epsilon 계산
    eps_gg = calculate_gg_epsilon(Y, current_weights)
    
    # 보정된 P-값
    p_gg = stats.f.sf(f_value, df_time * eps_gg, df_error * eps_gg) if df_error > 0 else np.nan

    # 5. Huynh-Feldt 보정
    eps_hf = calculate_hf_epsilon(eps_gg, effective_n, n_conditions)
    p_hf = stats.f.sf(f_value, df_time * eps_hf, df_error * eps_hf) if df_error > 0 else np.nan
    
    return {
        'F': f_value,
        'p_unc': p_unc, # Sphericity Assumed
        'p_gg': p_gg,
        'p_hf': p_hf,
        'eps_gg': eps_gg,
        'eps_hf': eps_hf,
        'm_w': m_w,
        'm_p': m_p,
        'df_model': df_time,
        'df_error': df_error,
        'weighted_n': current_sum_w if use_frequency_weight else np.sum(weights),
        'Means': mean_j.tolist()
    }
