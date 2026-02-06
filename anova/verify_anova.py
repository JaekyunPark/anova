import pandas as pd
import numpy as np
import pyreadstat
from run_anova import weighted_repeated_measures_anova

def create_dummy_sav(filename='dummy_data.sav'):
    np.random.seed(42)
    n = 100
    # Group 1 (Male) and Group 2 (Female)
    gender = np.random.choice([1, 2], size=n)
    
    # Weight: random weights between 0.5 and 1.5
    weights = np.random.uniform(0.5, 1.5, size=n)
    
    # Dependent variables: Time 1, Time 2, Time 3
    # Effect: Time 1 < Time 2 < Time 3
    t1 = np.random.normal(10, 2, size=n)
    t2 = np.random.normal(12, 2, size=n) # Significant increase
    t3 = np.random.normal(15, 2, size=n) # Further increase
    
    df = pd.DataFrame({
        'ID': range(n),
        'Gender': gender,
        'Weight': weights,
        'Time1': t1,
        'Time2': t2,
        'Time3': t3
    })
    
    # Add variable labels if possible (pyreadstat write_sav supports variable_value_labels)
    variable_value_labels = {
        'Gender': {1: 'Male', 2: 'Female'}
    }
    
    pyreadstat.write_sav(df, filename, variable_value_labels=variable_value_labels)
    print(f"Created {filename}")
    return df

def test_anova():
    df = create_dummy_sav()
    
    # Run ANOVA
    print("\nRunning Weighted RM ANOVA on Dummy Data...")
    dep_vars = ['Time1', 'Time2', 'Time3']
    res = weighted_repeated_measures_anova(df, dep_vars, 'Weight')
    
    print("Results:")
    print(f"F: {res['F']}")
    print(f"p: {res['p']}")
    print(f"Means: {res['Means']}")
    
    # Check simple validity
    if res['p'] < 0.05:
        print("SUCCESS: Detected significant effect as expected.")
    else:
        print("FAILURE: Did not detect significant effect.")
        
    # Check by banner
    print("\nRunning by Banner (Gender)...")
    for g in [1, 2]:
        sub_df = df[df['Gender'] == g]
        res_sub = weighted_repeated_measures_anova(sub_df, dep_vars, 'Weight')
        print(f"Gender {g}: F={res_sub['F']:.4f}, p={res_sub['p']:.4f}")

if __name__ == "__main__":
    test_anova()
