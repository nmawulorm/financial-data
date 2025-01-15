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

# TODO Step 3: Compare variances explained by each component
total_variance_explained = np.cumsum(explained_variance)  # Cumulative variance explained

for i, var in enumerate(explained_variance):
    print(f"Component {i+1} explains {var*100:.2f}% of the variance")

print("\nCumulative Variance Explained:")
print(total_variance_explained)


# TODO Step 4: Create a scree plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Variance Explained")
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid(True)
plt.show()


# TODO Conclusion:
# 1. Variance Analysis Explanation
# Here’s a possible paragraph summarizing the variance explained by each component:
#
# "The principal component analysis (PCA) results show that Component 1 explains 21.80% of the total variance, followed closely by Component 2 at 21.13%. Together, these two components account for approximately 43% of the variance in the data. Component 3 explains 19.74% of the variance, bringing the cumulative variance explained by the first three components to around 62.67%. The remaining components, Component 4 and Component 5, explain 18.94% and 18.39% of the variance, respectively. The relatively uniform distribution of explained variance among the components indicates that no single variable overwhelmingly dominates the dataset's variability, which is expected given that the original variables were generated to be uncorrelated."
#
# 2. Scree Plot Insights
# The scree plot demonstrates that the explained variance decreases gradually from Component 1 to Component 5. Unlike cases where a few components capture the majority of the variance, the variance here is spread more evenly across all five components. This distribution aligns with the characteristics of the data, as it was generated to have uncorrelated variables with similar variances.
#
# 3. Ethical and Practical Relevance
# Understanding the variance explained by components is crucial when deciding how many components to retain for further analysis. Retaining only the first few components might simplify the data while still capturing most of the variance. However, in this case, reducing dimensionality could lead to a loss of critical information because each component contributes significantly to the overall variance.
#
