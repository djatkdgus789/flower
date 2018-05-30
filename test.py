import random
import os
from dataset import Dataset
from model import CNN2D
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

batch_size = 32
image_shape = (80, 80, 3)
n_epochs = 100

# create dataset
data = Dataset(batch_size, image_shape)

# create model
model = load_model('./log/train/model_epochs_' + str(n_epochs) + '.h5')
model.summary()

# load weights
model.load_weights('./log/train/weights.hdf5')

# choice a random input
sample = random.choice(data.test)
print("\n-------------------------\n Sample:")
print(sample)
image = data.load_sample(sample)

# # output the true result
sample_result = data.get_class_one_hot(sample[0])
print("\n-------------------------\n True Result:")
print(sample_result)
print(sample_result.max())
print(sample_result.argmax())


# get predict result
predict_result = model.predict(image)[0]

print("\n-------------------------\n Predict Result:")
print(predict_result)
print(predict_result.max())
print(predict_result.argmax())
