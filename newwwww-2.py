#!/usr/bin/env python
# coding: utf-8

# # Importation des packetages

# In[48]:


import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import tensorflow as tf
import keras
import yfinance as yf
import numpy as np
from sklearn.preprocessing import StandardScaler , MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Input, Dropout, LSTM, Activation, Conv2D, Reshape, Average, Bidirectional
import plotly.graph_objects as go
from pandas_datareader import data as pdr
import plotly.express as px
from currency_converter import CurrencyConverter
from datetime import date


# # Collection des données

# In[49]:


tikers= ["BTC-USD"]


# In[50]:


data = yf.download(tikers, start="2018-07-07" ,end = "2020-07-07")


# In[51]:


btc = yf.Ticker("BTC-USD")


# In[52]:


data = btc.history(period="max")


# In[53]:


yf.pdr_override()


# In[54]:


df = pdr.get_data_yahoo("BTC-USD", period= "max")


# In[55]:


#Convertir les valeurs de dollar en euro 
c = CurrencyConverter()
for i in range(len(df['Close'])):
    df['Close'].iloc[i]=c.convert(df['Close'].iloc[i], 'USD', 'EUR')
Data=df
Data= Data.drop(['Open','High','Low','Adj Close','Volume'],axis = 1)


# # Visualisation des données 

# In[56]:


Data = Data.reset_index()   
Data.Date = pd.to_datetime(Data.Date)
Data.Date.sort_values().index
df_by_date = Data.iloc[Data.Date.sort_values().index]


# # La phase d'apprentissage

# In[58]:


#split the data into 80% training and 20%
x = Data.shape[0]
y = int(x*0.8)
v = x-y


# In[59]:


a=Data.drop(['Date'],axis = 1)


# In[60]:


data_training = a.head(y)


# In[61]:


#normalisation
scaler = MinMaxScaler()
training = scaler.fit_transform(data_training)


# In[62]:


def split(l, nbre_train, nbre_test):
    X, y = [], []
    for i in range(len(l)):
        end = i + nbre_train
        out_end = end + nbre_test
        if out_end > len(l):
            break
        seq_x, seq_y = l[i:end], l[end:out_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)  


# In[63]:


x_traing, y_traing = split(training, 100, 15) 


# In[64]:


#notre reseau de neuron 
regressior = Sequential()
regressior.add(LSTM(units=50,activation = 'relu',return_sequences = True , input_shape =( x_traing.shape[1],1)  ))
regressior.add(Dropout(0.2))
regressior.add(LSTM(units=60,activation = 'tanh',return_sequences = True ))
regressior.add(Dropout(0.2))
regressior.add(LSTM(units=80,activation = 'tanh',return_sequences = True ))
regressior.add(Dropout(0.2))
regressior.add(LSTM(units=180,activation = 'tanh',return_sequences = True ))
regressior.add(Dropout(0.2))
regressior.add(LSTM(units=120,activation = 'tanh' ))
regressior.add(Dropout(0.2))
regressior.add(Dense(units=15))


# In[65]:


regressior.summary()


# In[66]:


regressior.compile(optimizer='adam' , loss = 'mean_squared_error') #la fontion du perte et optimisation 


# In[67]:


regressior.fit(x_traing,y_traing,epochs = 20,batch_size = 32) 


# # La phase de test

# In[74]:


past_100_days = data_training.tail(100) #les dernières 100 lignes de data_training


# In[75]:


dff = past_100_days.append(a.tail(v)   , ignore_index = True)  #a.tail(v)=data_testing
# utiliser les dernières 100 lignes de data_training pour tester sur data_testing 


# In[76]:


inputs = scaler.transform(dff)


# In[77]:


x_test, y_test = split(inputs, 100, 15)


# In[78]:


y_pred = regressior.predict(x_test)


# In[79]:


scale = 1/scaler.scale_


# In[80]:


y_pred = y_pred*scale
y_test = y_test*scale


# In[81]:


Y_test = []
Y_pred = []

for i in range(414):
    for j in range(15):
        Y_test.append(y_test[i][j])
        Y_pred.append(y_pred[i][j])


# # Prédiction

# In[82]:


#prediction
l,v=split(scaler.transform(a.tail(100)), 100, 0)
c = regressior.predict(l) #utiliser les 100 dernières lignes (les 100 dernièrs jours ) pour prédire les 15 jours qui vont arriver 
scaler.scale_
scale = 1/scaler.scale_
H=c*scale


# In[83]:


from datetime import date

import datetime
d = {'prediction':H[0]}
DF = pd.DataFrame(d)

DF['Date'] = pd.date_range(start=datetime.date.today() + datetime.timedelta(days=1), periods=len(DF), freq='D')  


# # La partie de Streamlit 

# In[87]:


st.title("MY APP")


# In[88]:


st.write("""
#Le suivi et la prediction du BITCOIN
""")


# In[89]:


option = st.sidebar.selectbox('choisir ',('prediction','historique'))
if option == 'prediction':
    st.title('**le graphe**')
    fig2 = px.line(DF, x="Date", y="prediction",title='predict prices')
    ts_chart = st.plotly_chart(fig2)
    st.title('**Prediction**')
    i = st.sidebar.slider('selection du jour',0,14)
    data = [[DF['Date'].iloc[i], DF['prediction'].iloc[i]]]
    df = pd.DataFrame(data, columns = ['Date', 'prediction'])
    st.write(df)
if option == 'historique':
    st.title('**le graphe**')
    Data = Data.reset_index()   
    Data.Date = pd.to_datetime(Data.Date)
    Data.Date.sort_values().index
    df_by_date = Data.iloc[Data.Date.sort_values().index]
    st.write(Data)
    fig1 = px.line(Data, x="Date", y="Close",title='historical prices')
    ts_chart = st.plotly_chart(fig1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




