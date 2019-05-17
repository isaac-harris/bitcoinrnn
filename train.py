#from model import neuralnet

# image_data,y,image_test,text_y=tflearn.datasets.mnist.load_data(one_hot=True)
#
# image_input=image_data.reshape([-1,28,28,1])
# image_test=image_test.reshape([-1,28,28,1])

#neuralnet().fit({"input":image_input},{"targets":y},validation_set=(image_test,text_y),show_metric=True,n_epoch=10,shuffle=True,
#                run_id="test",snapshot_step=500)
import cv2
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

def bitcoin():
    import pandas
    import tflearn
    import numpy as np
    import cv2
    from tflearn.data_utils import to_categorical, pad_sequences
    from tflearn.datasets import imdb
    import tensorflow as tf


    data=pandas.read_csv("bitstampUSD_1-min_data_2012-01-01_to_2018-11-11.csv")
    df = pandas.DataFrame(data)
    train=data[:int(len(data)/2)].to_dict()
    test=data[int(len(data)/2):].to_dict()
    trainY,trainX=list(train["Timestamp"].values()),list(train["Weighted_Price"].values())
    testY,testX=list(test["Timestamp"].values()),list(test["Weighted_Price"].values())
    #time stamp is y val and weighted price is x

    # Converting labels to binary vectors
    # trainY = to_categorical(trainY,10000000000000000000000000000000)
    # testY = to_categorical(testY,1000000000000000000000000000000000)
    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]
    # testX=[x if (x!="NaN") else None for x in testX]
    # new_testx=[]
    # new_testy=[]
    # new_trainx = []
    # new_trainy = []
    # testx=chunks(testX,100)
    # for i in testx:
    #     new_testx.append(i)
    # testy=chunks(testY,100)
    # for j in testy:
    #     new_testy.append(j)
    # trainx=chunks(trainX,100)
    # trainy=chunks(trainY,100)
    # for k in trainx:
    #     print(k)
    #     new_trainx.append(k)
    # for l in trainy:
    #     new_trainy.append(l)
    #
    # np.reshape(trainX,[-1,100,1])
    # np.reshape(trainY, [-1, 100, 1])
    # np.reshape(testX, [-1, 100, 1])
    # np.reshape(testY, [-1, 100, 1])
    # print(new_trainx[:5])
    # print(new_trainy[:5])
    #print([x for x in new_trainx])
    net = tflearn.input_data([None,len(trainX)])
    net = tflearn.embedding(net, input_dim=10000, output_dim=128)
    net = tflearn.lstm(net, 128, dropout=0.8)
    net = tflearn.fully_connected(net, 2, activation='softmax')
    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                             loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net, tensorboard_verbose=0)

#        model.fit(np.array(new_trainx)[i], np.array(new_trainy)[i], validation_set=(np.array(new_testx)[i], np.array(new_testy)[i]), show_metric=True,batch_size=100)
    model.fit(np.array(trainX), np.array(trainY), validation_set=(np.array(testX), np.array(testY)), show_metric=True,batch_size=100)

#bitcoin()
