import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import datetime


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