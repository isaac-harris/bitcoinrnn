# by danhyal

import csv

import pandas
import numpy as np
from keras.engine.saving import load_model, model_from_json
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import torch
import matplotlib.pyplot as plt

import torch.nn as nn

def process_data():
    import tflearn
    import tensorflow as tf
    from tflearn.data_utils import to_categorical, pad_sequences
    csv_dict={}
    labels=[]
    comment=[]
    subreddit=[]
    ups=[]
    downs=[]
    parent_comment=[]
    with open("train-balanced-sarcasm.csv","r") as f:
        data=csv.DictReader(f)
        for i in data:
            labels.append(i["label"])
            comment.append(i["comment"])
            subreddit.append(i["subreddit"])
            ups.append(i["ups"])
            downs.append(i["downs"])
            parent_comment.append(i["parent_comment"])

    csv_dict.update({"label":labels,"comment":comment,"subreddit":subreddit,"ups":ups,
                     "downs":downs,"parent_comment":parent_comment})
    train_total=[list(csv_dict["label"])[int(len((csv_dict["label"]))/2):],

                 list(csv_dict["comment"])[0:int(len((csv_dict["comment"])) / 2):]]
    trainX,trainY=train_total[1],train_total[0]

    test_total=[list(csv_dict["label"])[0:int(len((csv_dict["label"]))/2)],

                list(csv_dict["comment"])[0:int(len((csv_dict["comment"]))/2)]]

    testX,testY=test_total[1],test_total[0]
    trainX = pad_sequences(trainX, maxlen=100, value=0.)
    testX = pad_sequences(testX, maxlen=100, value=0.)
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    network = tflearn.input_data(shape=[None, 100], name='input')
    network = tflearn.embedding(network, input_dim=10000, output_dim=128)
    branch1 = tflearn.conv_1d(network, 128, 3, padding='valid', activation='relu', regularizer="L2")
    branch2 = tflearn.conv_1d(network, 128, 4, padding='valid', activation='relu', regularizer="L2")
    branch3 = tflearn.conv_1d(network, 128, 5, padding='valid', activation='relu', regularizer="L2")
    network = tflearn.merge([branch1, branch2, branch3], mode='concat', axis=1)
    network = tf.expand_dims(network, 2)
    network = tflearn.layers.conv.global_max_pool(network)
    network = tflearn.dropout(network, 0.5)
    network = tflearn.fully_connected(network, 2, activation='softmax')
    network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy', name='target')
    # Training
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(testX, testY), show_metric=True, batch_size=32)

    #print(test)
    #print(trainx[5:10],trainy[5:10]

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
def bitcoin():




    data=pandas.read_csv("bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
    df = pandas.DataFrame(data)
    df["date"]=pandas.to_datetime(df["Timestamp"],unit="s").dt.date
    group=df.groupby("date")
    sorted_price=group["Weighted_Price"].mean()
    days=30
    scaler=MinMaxScaler()
    train=sorted_price[:len(sorted_price)-days]
    training_set=train.values
    training_set=np.reshape(training_set,(len(training_set),1))


    training_set=scaler.fit_transform(training_set)
    ## reshapes the input into [data size,1,1]
    X_train = training_set[0:len(training_set) - 1]
    y_train = torch.from_numpy(np.array(training_set[1:len(training_set)]))
    X_train = torch.from_numpy(np.reshape(X_train, (len(X_train), 1, 1)))

    ## trains model

    regressor = Sequential()

    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error',metrics=["accuracy"])
    regressor.fit(X_train, y_train, batch_size=5, epochs=200)
    # serialize model to JSON

    regressor.save("model.h5")

    print("Saved model to disk")



def process():
    ## loads data and sorts
    data = pandas.read_csv("bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
    df = pandas.DataFrame(data)
    df["date"] = pandas.to_datetime(df["Timestamp"], unit="s").dt.date
    group = df.groupby("date")
    sorted_price = group["Weighted_Price"].mean()
    days = 30
    ## normalizes data with min max transform
    scaler = MinMaxScaler()
    train = sorted_price[:len(sorted_price) - days]
    test = sorted_price[len(sorted_price) - days:]
    training_set = train.values
    training_set = np.reshape(training_set, (len(training_set), 1))
    training_set=scaler.fit_transform(training_set)
    model=load_model("model.h5")
    test_set = test.values
    test_set = np.reshape(test_set, (len(test_set), 1))
    test_set = scaler.transform(test_set)
    test_set = np.reshape(test_set, (len(test_set), 1, 1))
    ## gets prediction using test data
    prediction = model.predict(test_set)
    predicted_price = np.array(scaler.inverse_transform(prediction))
    test_set=test_set.reshape([30,1])
    test_set=scaler.inverse_transform(test_set)
    print(test_set,predicted_price)
    ## displays data onto matplot
    plt.figure(figsize=(25,15),dpi=90,)
    plt.plot(test_set,color="red",label="real")
    plt.plot(predicted_price,color="blue",label="predicted")
    test=test.reset_index()
    x=test.index
    labels=test["date"]
    plt.xticks(x,labels,rotation="vertical")

    plt.xlabel('time', fontsize=40)
    plt.ylabel('btc price', fontsize=40)
    plt.legend(loc=2, prop={'size': 25})
    plt.show()
# process()
def badclass():
    class enemy:
        def __init__(self):
            self.weps=100
            self.health=50

    def sleep(object):
        object.health+=1

    enemy1=enemy()
    enemy1.health=100
    enemy1.weps=50
    print(" first enemy has health {}".format(enemy1.health))

    human=enemy()
    enemy1.health=50

    print(human.health)
    while human.health>0:
        dmga=int(input(print("enter dmg amount")))
        human.health-=dmga
        print(human.health)

    class enem(enemy):
        def __init__(self):
            super().__init__()
        def count_Wep(self):
            print(self.weps)
    class enem2(enemy):
        pass
    enemy2=enem()

    enemy2.weps=0
    enemy2.count_Wep()

#
# import scipy as sc
# print(sc.fft._backend.ua)
# while True:
#     mat=np.array(np.random.rand(10,10)).transpose()
#     import scipy.fft as sf
#     print(np.array(sf.fft(mat)).cumprod())
from cv2 import cv2

# this is just to unconfuse pycharm
try:
    from cv2 import cv2
except ImportError:
    pass
print(cv2)
import matplotlib as mpl
import opencv_wrapper.display
mpl.use("WXAgg")
import matplotlib.pyplot as plt
print()
import matplotlib.rcsetup as rcsetup
from matplotlib.ticker import StrMethodFormatter
img=cv2.imread("/home/danhyal/test.jpg")
opencv_wrapper.display.line(img,(1,2),color=2,point2=(2,1))
print(img)
df2=pandas.read_csv("bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
df=pandas.read_csv("/home/danhyal/knifecrime2.csv")
print(df.keys())
print(rcsetup.all_backends)
print(plt.get_backend())
ax = df.plot()
plt.show()
import sqlite3
# conn=sqlite3.connect("/home/danhyal/database.laccdb")
# c=conn.cursor()
# print(c.execute("SELECT *"))