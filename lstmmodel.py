from flask import Flask,render_template
from flask_jsonpify import jsonpify


import numpy as np
import pandas as pd
import pandas_datareader as pdr
import math
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
plt.style.use("fivethirtyeight")

#tiingo api key
#key="51ee0264aa4987587760e2811d57719947b7d879"

#data=pdr.get_data_tiingo("AAPL",api_key=key)

#data.to_csv("AAPL1.csv")

df=pd.read_csv("AAPL1.csv")

# def set_df(ticker,api_key):
#     #ticker is the short name of the stock like AAPL
#     data = pdr.get_data_tiingo(ticker, api_key=api_key)
#     data.to_csv("AAPL1.csv")
#     df = pd.read_csv("AAPL1.csv")
#     return df
#
# df=set_df("AAPL",key)
print(df.head())


#Separate dates for future plotting
train_dates = pd.to_datetime(df['date'])

print(df.head(10))

def data_preprocessing(df):
    #geying only the important columns:
    cols=df.columns
    imp_cols=list(cols[2:7])
    #['close', 'high', 'low', 'open', 'volume']
    data=df.filter(imp_cols)
    return data

data=data_preprocessing(df)
print(data.head())


# for col in imp_cols:
#     plt.figure(figsize=(16,8))
#     plt.plot(df[col])
#     plt.xlabel("Date",fontsize=18)
#     plt.ylabel(col,fontsize=18)
#     plt.show()

#LSTM uses sigmoid and tanh that are sensitive to magnitude so values need to be normalized
# normalize the dataset
scaler = StandardScaler()
scaler = scaler.fit(data)
df_for_training_scaled = scaler.transform(data)


#As required for LSTM networks, we require to reshape an input data into n_samples x timesteps x n_features.
#In this example, the n_features is 2. We will make timesteps = 3.
#With this, the resultant n_samples is 5 (as the input data has 9 rows)

trainX = []
trainY = []

n_future = 1   # Number of days we want to predict into the future
n_past = 14     # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:data.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 3])#the 3 is the index of open column

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

model = keras.models.load_model('model.h5')
print(model.summary())

# fit model
history = model.fit(trainX, trainY, epochs=10, batch_size=16, validation_split=0.1, verbose=1)

#Forecasting...
#Start with the last day in training date and predict future...
n_future=30  #Redefining n_future to extend prediction dates beyond original n_future dates...
forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

forecast = model.predict(trainX[-n_future:]) #forecast

#Perform inverse transformation to rescale back to original range
#Since we used 5 variables for transform, the inverse expects same dimensions
#Therefore, let us copy our values 5 times and discard them after inverse transform
forecast_copies = np.repeat(forecast, data.shape[1], axis=-1)
y_pred_future = scaler.inverse_transform(forecast_copies)[:,0]

print(y_pred_future)

# Convert timestamp to date
forecast_dates = []
for time_i in forecast_period_dates:
    forecast_dates.append(time_i.date())

df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

print(df_forecast)

# original = df[['date', 'open']]
# original['date']=pd.to_datetime(original['date'])
# original = original.loc[original['date'] >= '2020-5-1']
#
# sns.lineplot(original['date'], original['open'])
#sns.lineplot(df_forecast['Date'], df_forecast['Open']).plot()

# plt.figure(figsize=(16,8))
# plt.plot(df_forecast['Date'],df_forecast['Open'])
# plt.xlabel("Date",fontsize=18)
# plt.ylabel("Open",fontsize=18)
# plt.show()

app = Flask(__name__)

@app.route('/')
def get_op():
    #return str(df_forecast["Open"][0])
    df_list = df_forecast.values.tolist()
    JSONP_data = jsonpify(df_list)
    return JSONP_data

if __name__=="__main__":
    app.run(debug=True,port=8000)