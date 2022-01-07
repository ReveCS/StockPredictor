import datetime
import math
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


def get_price_by_ticker(symbol, startDate, endDate, timeInterval):
    stockInfo = YahooFinancials(symbol)
    data = stockInfo.get_historical_price_data(startDate, endDate, timeInterval)
    priceDataFrame = pd.DataFrame(data[symbol]['prices'])
    priceDataFrame = priceDataFrame.drop('date', axis=1).set_index('formatted_date')
    return (priceDataFrame)

# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    x_data, y_data = [], []
    for i in range(len(dataset) - time_step - 1):
        x_data.append(dataset[i:(i + time_step), 0])
        y_data.append(dataset[i + time_step, 0])
    return np.array(x_data), np.array(y_data)


from_date = datetime.date(2018, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.date.today().strftime("%Y-%m-%d")
time_interval = 'daily'
symbol = 'AAPL'

prices = get_price_by_ticker(symbol, from_date, end_date, time_interval)
print(prices)

# Save and transform the 'close price' column values
column = prices.loc[:, 'close'].values
dataset = column.reshape(-1, 1)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# Split data into train (80%) and test (20%) sets
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)
print(len(train_set))
print(len(test_set))

# Reshape into X=t and Y=t+1, timestep 240
x_train, y_train = create_dataset(train_set, 240)
x_test, y_test = create_dataset(test_set, 240)

prices['close'].plot(title='{} Stock Price'.format(symbol))
plt.show()

# Define the univariate LSTM model
# model = Sequential()
# model.add(LSTM(50, ))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')