import pandas as pd
import numpy as np
import os
from configuration import clstm_config
from sklearn import preprocessing

class DataLoader():
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.data_file_path = config.data_file_path
        self.padding = config.padding                # the gap between the windows
        self.num_classes = config.num_classes
        self.data_window = config.data_window        # the length of every window
        self.is_shuffle = config.is_shuffle
        self.data = pd.DataFrame()
        self.data_stream = []
        self.labels = []
        self.sequences = []
        self.x = []
        self.y = []
        self.num_batches = 0
        self.load_data()


    def load_data(self):
        for i in range(1, 68):
            pathname = os.path.join(self.data_file_path, "real_"+str(i)+".csv")
            tmp = pd.read_csv(pathname)
            self.data = pd.concat([self.data, tmp])
        self.data = self.data.reset_index()
        self.data = self.data.drop("timestamp", axis=1)
        minx = self.data["value"].min()
        maxx = self.data["value"].max()
        norm = self.data["value"].apply(lambda x: float(x - minx) / (maxx - minx))
        self.data = self.data.drop("value", axis=1)
        self.data["value"] = norm

        for i in range(self.data.shape[0]):
            features = []
            features.append(self.data["value"][i])
            self.data_stream.append(features)
            self.labels.append(self.data["is_anomaly"][i])

        for i in range(0, self.data.shape[0]-self.data_window+1, self.padding):
            sequence = []
            anom = 0
            for j in range(self.data_window):
                sequence.append(self.data_stream[j])
                if (self.labels[i * self.padding + j] == 1):
                    anom = 1
            self.x.append(sequence)
            self.y.append([anom])
        self.x = np.array(self.x)
        self.y = np.array(self.y)
    
        if self.is_shuffle:
            self.shuffle()

        # split the dataset into training set and testing set
        self.x_train = self.x[0 : 66368, :]
        self.y_train = self.y[0 : 66368, :]

        self.x_test = self.x[66368:, :]
        self.y_test = self.y[66368:, :]

        self.num_batches_train = int(self.x_train.shape[0]) / self.batch_size
        self.num_batches_test = int(self.x_test.shape[0]) / self.batch_size
        self.x_train = self.x_train[:self.num_batches_train * self.batch_size]
        self.y_train = self.y_train[:self.num_batches_train * self.batch_size]
        self.x_test = self.x_test[:self.num_batches_test * self.batch_size]
        self.y_test = self.y_test[:self.num_batches_test * self.batch_size]
        self.sequence_batch_x_train = np.split(self.x_train, self.num_batches_train, 0)
        self.sequence_batch_y_train = np.split(self.y_train, self.num_batches_train, 0)
        self.pointer = 0
        self.num_batches = self.num_batches_train

    def shuffle(self):
        state = np.random.get_state()
        np.random.shuffle(self.x)
        np.random.set_state(state)
        np.random.shuffle(self.y)

    def next_batch(self):
        # Return numpy arrays of size:
        # train_x [self.batch_size, self.data_window, self.num_classes]
        # train_y [self.batch_size, label]
        train_x = self.sequence_batch_x_train[self.pointer]
        train_y = self.sequence_batch_y_train[self.pointer]
        train_y = train_y.reshape([train_y.shape[0], ])
        self.pointer = (self.pointer + 1) % self.num_batches_train
        return train_x, train_y

    def get_test_data(self):
        self.y_test = self.y_test.reshape([self.y_test.shape[0], ])
        return self.x_test, self.y_test

    def reset_batch(self):
        self.pointer = 0
"""
if __name__=="__main__":
    config = clstm_config()
    dataloader = DataLoader(config)
    tmp = dataloader.next_batch()
    hhh = dataloader.get_test_data()
    print (tmp[0].shape)
    print (tmp[1].shape)
    print (hhh[0].shape)
    print (hhh[1].shape)
    print (dataloader.num_batches)

    a = dataloader.y_train
    print (a[a==1].shape)
    """
