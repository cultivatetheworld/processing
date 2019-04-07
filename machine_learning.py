import tensorflow as tf
from pandas import read_csv
from datetime import datetime
from pandas import read_csv
from matplotlib import pyplot

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import math
from datetime import datetime
# load data
def parse(x):
  a =x.split("/")
  b= (datetime(int(a[2]),int(a[0]),int(a[1])))
  return b.strftime("%m/%d/%Y")
# dataset = read_csv('rdu-weather-history (1).csv',  parse_dates = [["date"]], index_col=0, date_parser=parse)
# dataset.drop('No', axis=1, inplace=True)
# manually specify column names
# dataset.columns = ['tempmin', 'tempmax', 'precip', 'press', 'avgwindspeed', 'fog', 'mist', 'rain', 'snow', 'freezingrain']
# dataset.index.name = 'date'
# mark all NA values with 0
# dataset['data'].fillna(0, inplace=True)
# drop the first 24 hours
# dataset = dataset[24:]
# summarize first 5 rows
# print(dataset.head(5))
# save to file
# dataset.to_csv('weather.csv')
tot_days = 365
# load dataset
dataset = read_csv('rdu-weather-history (1).csv', header=0, index_col=0)
values = dataset.values
# specify columns to plot
groups = [0, 1, 2, 3, 5, 6, 7, 8]
i = 1
n_hours = 4474
n_features = 9
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# load dataset
dataset = read_csv('rdu-weather-history (1).csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
from itertools import filterfalse
values = [x for x in values if len(x)==9]
for i in range (0, len(values)):
  for p in range(0, len(values[i])):
    if(values[i][p]=='Yes'):
      values[i][p]=1
    elif(values[i][p]=='No'):
      values[i][p]=0
values = np.array(values)
values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train_days = math.floor(tot_days * 0.8)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# make a prediction
print("Input:")
for i in test_X:
  print (i)
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
# invert scaling for forecast
test_X = test_X.reshape((test_X.shape[0], 10))
inv_yhat = concatenate((yhat, test_X[:,-8:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0:2] #changed from 0 to 0:2. Should be first 2 columns that contain the predictions
print ("Output:", inv_yhat)
#CHANGES HERE
#invert scaling for actual
test_y = test_y.reshape((len(test_y),1)) #changed 1 to 2
inv_y = concatenate((test_y, test_X[:,-8:]), axis=1) #changed 7 to 6
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0:2] #changed from 0 to 0:2. Should be first 2 columns that contain the predictions.
 
#CHANGES HERE
#calculate RMSE - CHANGED to output RMSE for each variable.
rmse_1 = sqrt(mean_squared_error(inv_y[:,0], inv_yhat[:,0])) #RMSE for the first variable (pollution)
rmse_2 = sqrt(mean_squared_error(inv_y[:,1], inv_yhat[:,0])) #RMSE for the second variable (dew)
print('Test RMSE: ', rmse_1, rmse_2)
