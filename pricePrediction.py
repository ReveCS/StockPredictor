import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from yahoofinancials import YahooFinancials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.core import Dense, Dropout


# Download historical daily closing price from Yahoo Finance
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


# Select stock and range of time for dataset
from_date = datetime.date(2018, 1, 1).strftime("%Y-%m-%d")
end_date = datetime.date.today().strftime("%Y-%m-%d")
time_interval = 'daily'
symbol = 'AAPL'

prices = get_price_by_ticker(symbol, from_date, end_date, time_interval)
#print(prices['adjclose'])
#print(prices)
#print("Null Value Present: ", prices.isnull().values.any())
#print(prices.info())

prices['close'].plot(title='{} Stock Price'.format(symbol))
#plt.show()

# Save and transform the 'adjusted close price' column values
column = prices.loc[:, 'adjclose'].values
predict_var = column.reshape(-1, 1)

# Select attributes
attributes = ['high', 'low', 'open', 'volume']

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
attribute_set = scaler.fit_transform(prices[attributes])
attribute_set = pd.DataFrame(columns=attributes, data=attribute_set, index=prices.index)
#print("dataset:", attribute_set)

# Split data into train (80%) and test (20%) sets
x_train, x_test, y_train, y_test = train_test_split(attribute_set, predict_var, test_size=0.2, shuffle=False)

# Reshape into X=t and Y=t+1, timestep 240
#x_train, y_train = create_dataset(train_set, 240)
#print("x_train:", x_train)
#print("x_test:", x_test)
#print("y-train:", y_train)
#print("y-test:", y_test)
#x_test, y_test = create_dataset(test_set, 240)

# Process the data and reshape input to be [samples, time steps, features]
trainX = np.array(x_train)
testX = np.array(x_test)
x_train = trainX.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = testX.reshape(x_test.shape[0], 1, x_test.shape[1])

# Define the univariate LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(1, trainX.shape[1])))
model.add(Dense(4))

# Configure the model with losses and metrics
model.compile(optimizer='adam', loss='mse')

# Train the model for 100 epochs with a batch size of 8
model.fit(x_train, y_train, epochs=250, batch_size=8, verbose=1, shuffle=False)

# Make and invert predictions
test_predict = model.predict(x_test)
#test_predict = scaler.inverse_transform(test_predict)

# Plot actual vs predicted values
plt.plot(y_test, label='True Value')
plt.plot(test_predict, label='LSTM Value')
plt.title("Prediction by LSTM")
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()
