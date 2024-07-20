import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_images(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, label)):
            img_path = os.path.join(data_dir, label, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

data_dir = 'data/train'
images, labels = load_images(data_dir)
x_train, x_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
