import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA

Number_of_features = 50


# Read Data
data = pd.read_csv('AAPL.csv')
data = data["Close"]

# Calculate steps
def cal_step(total_set, set, days):
    return int((total_set-set)/days)

#train and prediction
def train_and_predict(train_data, prediction_days):

    # Train Data
    model = ARIMA(train_data, order=(1, 0, 0))
    model_fit = model.fit()

    # Predict Data
    predictions = model_fit.predict(
        start=len(train_data), end=len(train_data)+prediction_days-1)

    return predictions


# valriables

total_set = len(data)
train_size = 500
days = 30
all_r2 = []


steps = cal_step(total_set, train_size, days)
iterated = 0

for step in range(steps):
    last_index = train_size + iterated

    close_data = data[iterated:last_index]
    actual_data = data[last_index:last_index+days]  # A/C to days
    iterated += days

    predicted = train_and_predict(close_data, days)

    # Calculate R Square
    r2 = r2_score(actual_data,predicted)

    all_r2.append(r2)

print(all_r2)