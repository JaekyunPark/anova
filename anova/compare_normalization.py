import pandas as pd
import numpy as np
import scipy.stats as stats

def anova_with_logic(data, dep_vars, weights, normalize=True):
    n_samples = len(data)
    n_conditions = len(dep_vars)
    weighted_n = np.sum(weights)

    if normalize:
        # Normalization: sum(norm_weights) == n_samples
        norm_weights = weights * n_samples / weighted_n
    else:
        # No normalization: raw weights
        norm_weights = weights

    # Actual sum of weights used in SS calculations
    current_sum_w = np.sum(norm_weights)

    Y = data[dep_vars].values
    mean_j = np.average(Y, axis=0, weights=norm_weights)
    grand_mean = np.average(Y, weights=np.tile(norm_weights[:, None], (1, n_conditions)))

    # SS_Total
    ss_total = np.sum(norm_weights[:, None] * (Y - grand_mean)**2)
    
    # SS_Subjects
    subject_means = np.mean(Y, axis=1)
    ss_subjects = np.sum(norm_weights * n_conditions * (subject_means - grand_mean)**2)
    
    # SS_Time
    # Using current_sum_w instead of n_samples to be mathematically consistent with norm_weights
    ss_time = current_sum_w * np.sum((mean_j - grand_mean)**2)
    
    # SS_Error
    ss_error = ss_total - ss_subjects - ss_time
    
    # Degrees of Freedom
    df_time = n_conditions - 1
    
    if normalize:
        df_error = (n_samples - 1) * (n_conditions - 1)
    else:
        # If not normalizing, what should df_error be?
        # If we use n_samples: F-value remains identical to normalized case, but SS values are scaled.
        # If we use weighted_n: p-values change drastically.
        df_error = (n_samples - 1) * (n_conditions - 1)
        # df_error_weighted = (weighted_n - 1) * (n_conditions - 1)

    ms_time = ss_time / df_time
    ms_error = ss_error / df_error
    
    f_value = ms_time / ms_error
    p_value = stats.f.sf(f_value, df_time, df_error)
    
    return {
        'F': f_value,
        'p': p_value,
        'SS_Time': ss_time,
        'SS_Error': ss_error,
        'sum_w': current_sum_w,
        'df_error': df_error
    }

# Setup dummy data
np.random.seed(42)
n = 10
dep_vars = ['T1', 'T2', 'T3']
df = pd.DataFrame({
    'T1': np.random.normal(10, 2, n),
    'T2': np.random.normal(12, 2, n),
    'T3': np.random.normal(15, 2, n),
})
# Imagine weights that sum to a very large number (e.g. 1000)
weights = np.random.uniform(50, 150, n) 

print(f"Sample size (N): {n}")
print(f"Raw Weight sum: {np.sum(weights):.2f}\n")

res_norm = anova_with_logic(df, dep_vars, weights, normalize=True)
res_raw = anova_with_logic(df, dep_vars, weights, normalize=False)

print("--- Comparison Results ---")
print(f"{'Metric':<15} | {'Normalized':<15} | {'Raw Weights':<15}")
print("-" * 50)
print(f"{'Sum of Weights':<15} | {res_norm['sum_w']:<15.2f} | {res_raw['sum_w']:<15.2f}")
print(f"{'SS_Time':<15} | {res_norm['SS_Time']:<15.4f} | {res_raw['SS_Time']:<15.4f}")
print(f"{'SS_Error':<15} | {res_norm['SS_Error']:<15.4f} | {res_raw['SS_Error']:<15.4f}")
print(f"{'F-Value':<15} | {res_norm['F']:<15.4f} | {res_raw['F']:<15.4f}")
print(f"{'DF_Error':<15} | {res_norm['df_error']:<15.0f} | {res_raw['df_error']:<15.0f}")
print(f"{'P-Value':<15} | {res_norm['p']:<15.4f} | {res_raw['p']:<15.4f}")

# Bonus: what if we used weighted_n in DF?
df_err_weighted = (np.sum(weights) - 1) * (len(dep_vars) - 1)
ms_time_raw = res_raw['SS_Time'] / (len(dep_vars) - 1)
ms_error_raw_weighted = res_raw['SS_Error'] / df_err_weighted
f_weighted = ms_time_raw / ms_error_raw_weighted
p_weighted = stats.f.sf(f_weighted, (len(dep_vars) - 1), df_err_weighted)

print("\n--- If we also used weighted_n for Degrees of Freedom ---")
print(f"Weighted DF_Error: {df_err_weighted:.2f}")
print(f"P-Value (Weighted DF): {p_weighted:.6f}")
