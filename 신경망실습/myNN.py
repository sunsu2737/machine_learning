import tensorflow as tf
from tensorflow import keras
import numpy as np
import HandleInput
from matplotlib import pyplot as plt
IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5
class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
train_features, train_labels, test_features, test_labels = HandleInput.load_all_data()

plt.figure()
plt.imshow(train_features[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_features[i])
    plt.xlabel(class_names[train_labels[i]])
plt.show()


def create_model():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
        keras.layers.Dense(512, activation=keras.activations.relu),
        keras.layers.Dense(NUM_CLASS, activation=keras.activations.softmax)
    ])

    model.compile(optimizer=tf.optimizers.Adam(
    ), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
    return model

check_point='traning/cp-{epoch:04d}.ckpt'

cp_callback = keras.callbacks.ModelCheckpoint(check_point,save_weights_only=True,verbose=1,period=5)

model = create_model()
model.summary()
# model.fit(train_features, train_labels, epochs=100,
#           validation_data=(test_features, test_labels),
#           callbacks=[cp_callback])

model.load_weights('traning/cp-0100.ckpt')

test_loss, test_acc = model.evaluate(test_features, test_labels)
print('Test accuracy', test_acc)

predictions = model.predict(test_features)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label], 100 * np.max(
        predictions_array), class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_features)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_features)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions,  test_labels)
plt.show()


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_features)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
img = test_features[0]
print(img.shape)
# Add the image to a batch where it's the only member.
img = (np.expand_dims(img, 0))
print(img.shape)
predictions_single = model.predict(img)
print(predictions_single)
plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(5), class_names, rotation=45)
plt.show()
