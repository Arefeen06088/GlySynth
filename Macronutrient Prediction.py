import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sklearn
import math
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from keras.callbacks import LambdaCallback
from sklearn import utils
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dropout, Attention, Activation, MaxPooling1D, Flatten, Dense, LSTM, Conv1D, Flatten, AveragePooling1D, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from attention import Attention

def moving_avarage_smoothing(X,k):
    S = np.zeros(X.shape[0])
    for t in range(X.shape[0]):
        if t < k:
            S[t] = np.mean(X[:t+1])
        else:
            S[t] = np.sum(X[t-k:t])/k
    return S

def load_macronutrients (data_file, subject_no, data_type):
    csv = pd.read_excel(data_file, subject_no, skiprows = 0)
    if (data_type=='real'):
        nutrients = csv[csv.columns[1:4]].to_numpy()
    if (data_type=='synthetic'):
        nutrients = csv[csv.columns[2:5]].to_numpy()
    return nutrients

def load_glycemic_response (data_file, subject_no, data_type):
    csv = pd.read_excel(data_file, subject_no, skiprows = 0)
    if (data_type=='real'):
        cgm = csv[csv.columns[9:129]].to_numpy()
    if (data_type=='synthetic'):
        cgm = csv[csv.columns[7:127]].to_numpy()
    print(cgm.shape)
    return cgm

def call_model():
    model_input = Input(shape=(train_x.shape[1], train_x.shape[2]))

    x = Conv1D(10, kernel_size=1, kernel_regularizer=l2(0.06), activation='relu')(model_input)
    x = Dropout(0.5)(x)

    x = LSTM(4, kernel_regularizer=l2(0.06), activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(units = 5)(x)

    carb = Dense(units = 1, activation='linear', name='carb')(x)
    fat = Dense(units = 1, activation='linear', name='fat')(x)
    fiber = Dense(units = 1, activation='linear', name='fiber')(x)

    model = Model(inputs=model_input, outputs=[carb,fat,fiber])

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-05)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
    
    return model

def fit_the_model(model,train_x,train_y,test_x,test_y):
    train_carb,train_fat,train_fiber,test_carb,test_fat,test_fiber = train_y[:,0],train_y[:,1],train_y[:,2],test_y[:,0],test_y[:,1],test_y[:,2]
    h_model = model.fit(
        train_x, (train_carb,train_fat,train_fiber),
        epochs = 1700,
        batch_size = 4,
        validation_data = (test_x, (test_carb,test_fat,test_fiber)),
        verbose = True,
        shuffle = False
    )
    return model

xlsx = pd.ExcelFile('customCGM.xlsx')
xs = pd.ExcelFile('Synthetic data\Generated CGMs -- USDA P4.xlsx')

P01_ = load_macronutrients (xlsx, 'P01', 'real')
P01 = load_glycemic_response (xlsx, 'P01', 'real')

P02_ = load_macronutrients (xlsx, 'P02', 'real')
P02 = load_glycemic_response (xlsx, 'P02', 'real')

P03_ = load_macronutrients (xlsx, 'P03', 'real')
P03 = load_glycemic_response (xlsx, 'P03', 'real')

P04_ = load_macronutrients (xlsx, 'P04', 'real')
P04 = load_glycemic_response (xlsx, 'P04', 'real')

P05_ = load_macronutrients (xlsx, 'P05', 'real')
P05 = load_glycemic_response (xlsx, 'P05', 'real')

synth_y = load_macronutrients (xs, 'Generated CGMs -- USDA P4', 'synthetic')
synth_x = load_glycemic_response (xs, 'Generated CGMs -- USDA P4', 'synthetic')

sub = P03
sub_ = P03_
sub, sub_ = shuffle(sub, sub_, random_state=13)
CHO = []
Fat = []
Fiber = []
for i in range (0,5):
    test_x = sub[i*int(sub.shape[0]/5):(i+1)*int(sub.shape[0]/5)]
    test_y = sub_[i*int(sub.shape[0]/5):(i+1)*int(sub.shape[0]/5)]
    train_x = np.delete(sub, (np.linspace(i*int(sub.shape[0]/5),(i+1)*int(sub.shape[0]/5)-1,5)).astype(int),axis=0)
    train_y = np.delete(sub_, np.linspace(i*int(sub.shape[0]/5),(i+1)*int(sub.shape[0]/5)-1,5).astype(int),axis=0)
    train_x = (np.concatenate((train_x, synth_x),axis=0))
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    train_y = (np.concatenate((train_y, synth_y),axis=0))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    model = call_model()
    model = fit_the_model(model,train_x,train_y,test_x,test_y)
    pred = (model.predict(test_x)[0]).reshape((model.predict(test_x)[0]).shape[0],)
    CHO.append(math.sqrt(np.mean(((test_y[:,0]-pred)**2)/test_y[:,0]**2)))
    pred = (model.predict(test_x)[1]).reshape((model.predict(test_x)[1]).shape[0],)
    Fat.append(math.sqrt(np.mean(((test_y[:,1]-pred)**2)/test_y[:,1]**2)))
    pred = (model.predict(test_x)[2]).reshape((model.predict(test_x)[2]).shape[0],)
    Fiber.append(math.sqrt(np.mean(((test_y[:,2]-pred)**2)/test_y[:,2]**2)))
    print(np.mean(CHO),np.mean(Fat),np.mean(Fiber))

print(CHO)

print(Fat)

print(Fiber) 





