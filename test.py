import random
import os
from dataset import Dataset
from model import CNN2D
from keras.models import load_model
from keras.optimizers import SGD
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

batch_size = 16
image_shape = (224, 224, 3)
n_epochs = 100

# create dataset
data = Dataset(batch_size, image_shape)

# create model
model = load_model('./log/train/model_epochs_' + str(n_epochs) + '.h5')
model.summary()

# load weights
model.load_weights('./log/train/weights.h5')

# choice a random input
sample = random.choice(data.test)
print("\n-------------------------\n Sample:")
print(sample)
image = data.load_sample(sample)

# output the true result
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

train_dataset_count = 0
train_true_count = 0

for sample in data.train:
    image = data.load_sample(sample)
    sample_result = data.get_class_one_hot(sample[0])
    predict_result = model.predict(image)[0]
    if sample_result.argmax() == predict_result.argmax():
        train_true_count = train_true_count + 1
    train_dataset_count = train_dataset_count + 1
print(' Train accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(train_true_count, train_dataset_count, train_true_count / train_dataset_count))

test_dataset_count = 0
test_true_count = 0

for sample in data.test:
    image = data.load_sample(sample)
    sample_result = data.get_class_one_hot(sample[0])
    predict_result = model.predict(image)[0]
    if sample_result.argmax() == predict_result.argmax():
        test_true_count = test_true_count + 1
    test_dataset_count = test_dataset_count + 1

print(' Test accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(test_true_count, test_dataset_count, test_true_count / test_dataset_count))


true_count = train_true_count + test_true_count
dataset_count = train_dataset_count + test_dataset_count
print(' All accuary: {0:4d}  /  {1:4d}  =  {2:6f} \n'.format(true_count, dataset_count, true_count / dataset_count))

X_all = []
y_all = []
for sample in np.vstack((data.test, data.train)):
    X_all.append(data.load_sample(sample))
    y_all.append(data.get_class_one_hot(sample[0]).argmax())
X_all = np.array(X_all)
y_all = np.array(y_all)
Y_pred = []

for X in X_all:
    Y_pred.append(model.predict(X)[0].argmax())
y_pred = np.array(Y_pred)

FLOWER_NAME = ['개나리','나팔꽃','데이지','동자꽃','목화','백합','아부틸론','장미','해바라기']

classification = classification_report(y_all, y_pred, target_names=FLOWER_NAME)
confusion = confusion_matrix(y_all, y_pred)
# score = model.evaluate(X_all, y_all, batch_size=32)
# Test_Loss = score[0] * 100
# Test_accuracy = score[1] * 100
classification = str(classification)
confusion = str(confusion)

file_name = './report.txt'
with open(file_name, 'w') as x_file:
    # x_file.write('{} Test loss (%)'.format(Test_Loss))
    # x_file.write('\n')
    # x_file.write('{} Test accuracy (%)'.format(Test_accuracy))
    # x_file.write('\n')
    x_file.write('\n')
    x_file.write('{}'.format(classification))
    x_file.write('\n')
    x_file.write('{}'.format(confusion))