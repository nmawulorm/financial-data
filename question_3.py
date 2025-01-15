import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO Step 1: Generate 5 uncorrelated Gaussian random variables
np.random.seed(42)  # For reproducibility
num_variables = 5
num_samples = 1000

# Mean = 0, standard deviation = 0.1
random_data = np.random.normal(loc=0, scale=0.1, size=(num_samples, num_variables))

# Convert to a DataFrame for easier handling
df = pd.DataFrame(random_data, columns=[f"Var{i+1}" for i in range(num_variables)])
print(df.head())

# Verify that the variables are approximately uncorrelated
correlation_matrix = df.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

