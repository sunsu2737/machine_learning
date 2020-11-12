# import numpy as np
# table = np.genfromtxt('auto-mpg.data.csv', delimiter=',', skip_header=1)
# table=table[:,:-1]
# ind=np.isnan(table).any(axis=1)
# table=table[~ind,:]
# # print(table)
# target=table[:,-1].astype(np.int)
# values=np.unique(target)
# t=np.identity(values.shape[0])
# # print(t.shape)
# # print(t)

# encoded=t[target-1]
# # print(encoded)

# table = np.concatenate((table[:,:-1],encoded),axis=1)
# # print(table)
# np.random.shuffle(table)

# n=table.shape[0]
# train_dataset= table[0:int(0.8*n),:]
# test_data=table[int(0.8*n):,:]

# train_features = train_dataset[:, 1:]
# test_features = test_dataset[:, 1:]
# # print(train_features.shape)
# # print(test_features.shape)

# train_labels = train_dataset[:, 0]
# test_labels = test_dataset[:, 0]
# # print(train_labels.shape)
# # print(test_labels.shape)
########################################## numpy#########

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt

raw_dataset = pd.read_csv('insurance.csv',
                          na_values='?', sep=',', skipinitialspace=True, header=0)
dataset = raw_dataset.copy()
# print(dataset.tail())
# print(dataset.shape)
# print(dataset.isna().sum())
dataset = dataset.dropna()
# print(dataset.shape)
print(dataset.tail())


dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('charges')
test_labels = test_features.pop('charges')

train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
print(tf.__version__)
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                  optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
# %%time
history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=500)


def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Error [charge]')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss(history)

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [charge]')
plt.ylabel('Predictions [charge]')
lims = [0, 60000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()
error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [charge]')
_ = plt.ylabel('Count')
plt.show()
test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)
# dnn_model.save('dnn_model2')
