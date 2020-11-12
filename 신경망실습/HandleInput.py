
import numpy as np
import os
import cv2

IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3
NUM_CLASS = 5
IMAGE_DIR_BASE = 'animal_images'


def load_image(addr):
    img = cv2.imread(addr)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT),
                     interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - np.mean(img))/np.std(img)
    return img


def load_data_set(set_name):


    data_set_dir = os.path.join(IMAGE_DIR_BASE, set_name)
    image_dir_list = os.listdir(data_set_dir)
    image_dir_list.sort()
    features = []# empty Python list
    labels = []# 0 cat, 1 cow, 2 dog, 3 pig, 4 sheep
    for cls_index, dir_name in enumerate(image_dir_list):
        image_list = os.listdir(os.path.join(data_set_dir, dir_name))
        for file_name in image_list:
            if 'png' in file_name or 'jpg' in file_name or 'jpeg' in file_name:
                image = load_image(os.path.join(data_set_dir, dir_name, file_name))
                features.append(image)
                labels.append(cls_index)
    idxs = np.arange(0, len(features))
    np.random.shuffle(idxs)
    features = np.array(features)
    labels = np.array(labels)
    shuf_features = features[idxs]
    shuf_labels = labels[idxs]
    return shuf_features, shuf_labels

def load_all_data():
    train_images, train_labels = load_data_set('train')
    test_images, test_labels = load_data_set('test')
    return train_images, train_labels, test_images, test_labels
if __name__ =='__main__':
    train_images, train_labels, test_images, test_labels = load_all_data()
    print(train_images.shape)