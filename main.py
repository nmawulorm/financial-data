import pandas_datareader.data as web
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit




# Define the start and end dates for the data retrieval
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)

# List of FRED series IDs for different maturities
treasury_series = {
    '1 Month': 'DGS1MO',
    '3 Month': 'DGS3MO',
    '6 Month': 'DGS6MO',
    '1 Year': 'DGS1',
    '2 Year': 'DGS2',
    '3 Year': 'DGS3',
    '5 Year': 'DGS5',
    '7 Year': 'DGS7',
    '10 Year': 'DGS10',
    '20 Year': 'DGS20',
    '30 Year': 'DGS30'
}

# Fetch the data
treasury_data = pd.DataFrame()
for label, series in treasury_series.items():
    treasury_data[label] = web.DataReader(series, 'fred', start_date, end_date)

# Drop rows with missing values
treasury_data.dropna(inplace=True)

# Plot the yield curves for the first and last available dates
plt.figure(figsize=(12, 6))
plt.plot(treasury_series.keys(), treasury_data.iloc[0], marker='o', label=treasury_data.index[0].date())
plt.plot(treasury_series.keys(), treasury_data.iloc[-1], marker='o', label=treasury_data.index[-1].date())
plt.xlabel('Maturity')
plt.ylabel('Yield (%)')
plt.title('U.S. Treasury Yield Curve')
plt.legend()
plt.grid(True)
plt.show()

def nelson_siegel(maturity, beta0, beta1, beta2, lambda_):
    term1 = beta0
    term2 = beta1 * (1 - np.exp(-lambda_ * maturity)) / (lambda_ * maturity)
    term3 = beta2 * ((1 - np.exp(-lambda_ * maturity)) / (lambda_ * maturity) - np.exp(-lambda_ * maturity))
    return term1 + term2 + term3

maturities = [1/12, 3/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30]  # Convert months to years
yields = treasury_data.iloc[-1].values  # Use the latest yield data for fitting

# Initial guess for parameters
initial_guess = [4, -1, 1, 0.5]  # Adjust these as needed
params, _ = curve_fit(nelson_siegel, maturities, yields, p0=initial_guess)

print("Fitted Parameters (Beta0, Beta1, Beta2, Lambda):", params)
