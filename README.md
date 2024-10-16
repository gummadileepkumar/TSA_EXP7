#### Developed by: Gumma Dileep Kumar 
#### Reg no: 212222240032
#### Date:
## Ex.No: 07                                       AUTO REGRESSIVE MODEL
 



### AIM:
To Implementat an Auto Regressive Model using Python on National stocks exchange dataset
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/content/infy_stock-Copy1.csv')  

# Convert 'Date' to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Check for stationarity using the Augmented Dickey-Fuller (ADF) test on 'Volume'
result = adfuller(data['Volume']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

# Split data into training and testing sets (80% training, 20% testing)
train_data = data.iloc[:int(0.8 * len(data))]
test_data = data.iloc[int(0.8 * len(data)):]

# Define the lag order for the AutoRegressive model (adjust lag based on ACF/PACF plots)
lag_order = 13
model = AutoReg(train_data['Volume'], lags=lag_order)
model_fit = model.fit()

# Plot Autocorrelation Function (ACF) for 'Volume'
plt.figure(figsize=(10, 6))
plot_acf(data['Volume'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF) - Volume')
plt.show()

# Plot Partial Autocorrelation Function (PACF) for 'Volume'
plt.figure(figsize=(10, 6))
plot_pacf(data['Volume'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF) - Volume')
plt.show()

# Make predictions on the test set
predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)

# Calculate Mean Squared Error (MSE) for the test set predictions
mse = mean_squared_error(test_data['Volume'], predictions)
print('Mean Squared Error (MSE):', mse)

# Plot Test Data vs Predictions for 'Volume'
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Volume'], label='Test Data - Volume', color='blue', linewidth=2)
plt.plot(test_data.index, predictions, label='Predictions - Volume', color='orange', linestyle='--', linewidth=2)
plt.xlabel('Date')
plt.ylabel('Volume')
plt.title('AR Model Predictions vs Test Data (Volume)')
plt.legend()
plt.grid(True)
plt.show()
```
### OUTPUT:

### GIVEN DATA

![Screenshot 2024-10-16 154550](https://github.com/user-attachments/assets/4ae25a6f-420d-42bd-9830-830042fd9572)

### PACF - ACF
![Screenshot 2024-10-16 153928](https://github.com/user-attachments/assets/274d6535-a5aa-4587-885e-f84c445b7b7d)


### PREDICTION

![Screenshot 2024-10-16 153953](https://github.com/user-attachments/assets/7a7d05df-6025-4fcf-b585-a0ded3457d0e)

### FINIAL PREDICTION
![Screenshot 2024-10-16 154010](https://github.com/user-attachments/assets/2583b801-3bfa-4b56-bbbe-177ebb043f21)


### RESULT:
ThuS, successful implemention of the auto regression function using python is done.
