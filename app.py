from flask import Flask,render_template
from flask_jsonpify import jsonpify

import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64
import os
from tensorflow import keras
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
plt.style.use("fivethirtyeight")
#%matplotlib widget



app = Flask(__name__)

imgFolder=os.path.join('static','images')
app.config['UPLOAD_FOLDER']=imgFolder

#tiingo api key
key="51ee0264aa4987587760e2811d57719947b7d879"

#data=pdr.get_data_tiingo("AAPL",api_key=key)
data=pdr.get_data_tiingo("AAPL",start='Jan, 1, 1980',api_key=key)

data.to_csv("AAPL1.csv")

#main dataframe
df=pd.read_csv("AAPL1.csv")

#dataframe for stats:
df1=pd.read_csv("AAPL1.csv",index_col="date",parse_dates=True)
cols1=df1.columns
imp_cols1=list(cols1[1:6])
df1=df1.filter(imp_cols1)




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
n_past = 22     # Number of past days we want to use to predict the future

for i in range(n_past, len(df_for_training_scaled) - n_future +1):
    trainX.append(df_for_training_scaled[i - n_past:i, 0:data.shape[1]])
    trainY.append(df_for_training_scaled[i + n_future - 1:i + n_future, 3])#the 3 is the index of open column

trainX, trainY = np.array(trainX), np.array(trainY)

print('trainX shape == {}.'.format(trainX.shape))
print('trainY shape == {}.'.format(trainY.shape))

#model = keras.models.load_model('model2.h5')
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




@app.route('/')
def home_page():
    return render_template("index1.html")



# @app.route('/predict')
# def get_result():
#     df_list = df_forecast.values.tolist()
#     JSONP_data = jsonpify(df_list)
#     return JSONP_data

@app.route('/about')
def about_page():
    pic1=os.path.join(app.config['UPLOAD_FOLDER'],'LSTMA.png')
    pic2=os.path.join(app.config['UPLOAD_FOLDER'],'lstmcell.png')

    return render_template("about.html",lstm_a=pic1,lstm_cell=pic2)

@app.route('/contact')
def contact_page():
    return render_template("contact.html")

@app.route("/predict", methods=["GET"])
def prediction_page():
    
    # Generate plot
    fig = Figure(figsize=(14,8))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Price vs Dates")
    axis.set_xlabel("Dates")
    axis.set_ylabel("price")
    #axis.grid()
    axis.plot(df_forecast['Date'],df_forecast['Open'],"ro-")
    # plt.figure(figsize=(16,8))
    # plt.plot(df_forecast['Date'],df_forecast['Open'])
    # plt.xlabel("Date",fontsize=18)
    # plt.ylabel("Open",fontsize=18)

    
    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')

    #Generate the table
    df_copy=df_forecast.copy()
    df_copy.set_index("Date",inplace=True)
    df_copy.index.name=None
    
    return render_template("graph.html", image=pngImageB64String,table=df_copy.to_html())
    #return render_template("graph.html", image=pngImageB64String)


@app.route("/Stats",methods=['GET'])
def stats_page():
    fresh_df=df1.tail(30)
    top_30=fresh_df.iloc[::-1]


    #generate overall graph
    fig = Figure(figsize=(14,8))
    
    #open 
    axis1 = fig.add_subplot(2, 2, 1)
    axis1.plot(df1["open"],label="Open")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    axis1.set_xlabel("Dates")
    axis1.set_ylabel('Open')
    axis1.legend(loc='upper left')
    #close 
    axis2 = fig.add_subplot(2, 2, 2)
    axis2.plot(df1["close"],label="Close")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    axis2.set_xlabel("Dates")
    axis2.set_ylabel('Close')
    axis2.legend(loc='upper left')
    #high
    axis3 = fig.add_subplot(2, 2, 3)
    axis3.plot(df1["high"])
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    #axis1.set_xlabel("Dates")
    #axis1.set_ylabel('High')

    #low
    axis3 = fig.add_subplot(2, 2, 3)
    axis3.plot(df1["low"])
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    axis3.set_xlabel("Dates")
    axis3.set_ylabel('high/low')
    axis3.legend(["High","Low"],loc='upper left')

    #volume
    axis4 = fig.add_subplot(2, 2, 4)
    axis4.plot(df1["volume"],label="Volume")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    axis4.set_xlabel("Dates")
    axis4.set_ylabel('volume')
    axis4.legend(loc='upper left')


    # Convert plot to PNG image
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')


        #plt.show()

    #generate 30 days graphs
    fig1 = Figure(figsize=(14,8))
    
    #open 
    a1 = fig1.add_subplot(2, 2, 1)
    a1.plot(top_30["open"],label="Open")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    a1.set_xlabel("Dates")
    a1.set_ylabel('Open')
    a1.legend(loc='upper left')
    #close 
    a2 = fig1.add_subplot(2, 2, 2)
    a2.plot(top_30["close"],label="Close")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    a2.set_xlabel("Dates")
    a2.set_ylabel('Close')
    a2.legend(loc='upper left')
    #high
    a3 = fig1.add_subplot(2, 2, 3)
    a3.plot(top_30["high"])
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    #axis1.set_xlabel("Dates")
    #axis1.set_ylabel('High')

    #low
    a3 = fig1.add_subplot(2, 2, 3)
    a3.plot(top_30["low"])
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    a3.set_xlabel("Dates")
    a3.set_ylabel('high/low')
    a3.legend(["High","Low"],loc='upper left')

    #volume
    a4 = fig1.add_subplot(2, 2, 4)
    a4.plot(top_30["volume"],label="Volume")
    # axis.xlabel("Date",fontsize=18)
    # axis.ylabel(col,fontsize=18)
    a4.set_xlabel("Dates")
    a4.set_ylabel('volume')
    a4.legend(loc='upper left')

    fig1.autofmt_xdate()
    # t_graph=plt.plot(top_30["open"],label="Open")
    # t_graph.show

    # Convert plot to PNG image
    pngImage1 = io.BytesIO()
    FigureCanvas(fig1).print_png(pngImage1)
    
    # Encode PNG image to base64 string

    pngImageB64String1 = "data:image/png;base64,"
    pngImageB64String1 += base64.b64encode(pngImage1.getvalue()).decode('utf8')


    return render_template("Stats.html",image=pngImageB64String,table=top_30.to_html(),image1=pngImageB64String1)
    #return render_template("Stats.html",image=pngImageB64String,table=top_30.to_html())
   


if __name__=="__main__":
    app.run(debug=True,port=8000)