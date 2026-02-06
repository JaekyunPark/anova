import pandas as pd
import numpy as np
import anova_logic

# Setup dummy data
np.random.seed(42)
n = 10
dep_vars = ['T1', 'T2', 'T3']
df = pd.DataFrame({
    'T1': np.random.normal(10, 2, n),
    'T2': np.random.normal(12, 2, n),
    'T3': np.random.normal(15, 2, n),
})
# Population weights (10 cases -> 1000 population)
weights = np.random.uniform(50, 150, n) 
df['Weight'] = weights

print(f"Sample size (N): {n}")
print(f"Weight sum (W): {np.sum(weights):.2f}\n")

# Current default
res_std = anova_logic.weighted_repeated_measures_anova(df, dep_vars, 'Weight', normalize=True, use_weighted_df=False)
# Weighted DF option
res_wdf = anova_logic.weighted_repeated_measures_anova(df, dep_vars, 'Weight', normalize=True, use_weighted_df=True)

print("--- F-Value Comparison ---")
print(f"{'Option':<25} | {'F-Value':<15} | {'P-Value':<15} | {'DF_Error':<15}")
print("-" * 75)
print(f"{'Standard (Sample DF)':<25} | {res_std['F']:<15.4f} | {res_std['p']:<15.4f} | {res_std['df_error']:<15.0f}")
print(f"{'Weighted (Population DF)':<25} | {res_wdf['F']:<15.4f} | {res_wdf['p']:<15.4f} | {res_wdf['df_error']:<15.0f}")

ratio = res_wdf['F'] / res_std['F']
pop_ratio = np.sum(weights) / n
print(f"\nF-Value Ratio: {ratio:.4f}")
print(f"Weight/Sample Ratio: {pop_ratio:.4f}")
