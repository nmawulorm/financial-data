import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
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


# TODO Perform Principal Component Analysis (PCA)
# Step 2: Run PCA using the covariance matrix
pca = PCA()
pca.fit(df)

# Extract explained variance and components
explained_variance = pca.explained_variance_ratio_  # Variance explained by each component
components = pca.components_  # Principal components

print("\nExplained Variance Ratio (by Component):")
print(explained_variance)

print("\nPrincipal Components:")
print(components)
