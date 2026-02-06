import pandas as pd
import numpy as np
import anova_logic

# Setup dummy data that violates sphericity
np.random.seed(42)
n = 20
# Two variables are highly correlated, one is not
t1 = np.random.normal(10, 2, n)
t2 = t1 + np.random.normal(0, 0.5, n) # Highly correlated with t1
t3 = np.random.normal(15, 5, n)        # Different variance/correlation

df = pd.DataFrame({
    'T1': t1,
    'T2': t2,
    'T3': t3,
    'Weight': np.random.uniform(0.5, 1.5, n)
})

print("--- Greenhouse-Geisser Verification ---")
res = anova_logic.weighted_repeated_measures_anova(df, ['T1', 'T2', 'T3'], 'Weight')

print(f"F-Value: {res['F']:.4f}")
print(f"Epsilon(ε): {res['epsilon']:.4f}")
print(f"Uncorrected P: {res['p']:.6f}")
print(f"Corrected P (GG): {res['p_gg']:.6f}")

if res['epsilon'] < 1.0:
    print("\nSUCCESS: Epsilon detected sphericity violation (< 1.0).")
else:
    print("\nFAILURE: Epsilon should be < 1.0 for this data.")

if res['p_gg'] > res['p']:
    print("SUCCESS: Corrected P is more conservative (larger) than uncorrected P.")
else:
    print("FAILURE: Corrected P should be >= uncorrected P.")
