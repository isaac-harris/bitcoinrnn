# by danhyal

import csv



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
    import pandas
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    import torch
    import torch.nn as nn



    data=pandas.read_csv("bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv")
    df = pandas.DataFrame(data)
    df["date"]=pandas.to_datetime(df["Timestamp"],unit="s").dt.date
    group=df.groupby("date")
    sorted_price=group["Weighted_Price"].mean()
    days=30
    train=sorted_price[:len(sorted_price)-days]
    test=sorted_price[len(sorted_price)-days:]
    training_set=train.values
    training_set=np.reshape(training_set,(len(training_set),1))


    training_set=MinMaxScaler().fit_transform(training_set)
    X_train = training_set[0:len(training_set) - 1]
    y_train = torch.from_numpy(np.array(training_set[1:len(training_set)]))
    X_train = torch.from_numpy(np.reshape(X_train, (len(X_train), 1, 1)))
    print(X_train.shape)



    regressor = Sequential()

    regressor.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    regressor.add(Dense(units=1))
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    regressor.fit(X_train, y_train, batch_size=5, epochs=100)
    print(regressor.outputs)

bitcoin()
