import os
import csv

from glob import glob
from sklearn.model_selection import train_test_split


FLOWERS_IMAGES = './data/'




def main():
    data = []

    # folder명을 label로, 각 folder안의 image matching
    for root, dirs, files in os.walk(FLOWERS_IMAGES):
        for file in files:
            if file == '.DS_Store':
                continue
            path = os.path.join(root, file)
    
            path_split = (path.split('/'))

            data.append((path_split[2], path))
            print(data)
        

    # data를 train, test dataset으로 나눈다.
    train, test = train_test_split(data, test_size=0.3, random_state=0)

    # create train dataset
    with open('./data/train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train)

    # create test dataset
    with open('./data/test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test)


if __name__ == '__main__':
    print('Creating csv file..........')
    main()
    print('Finished csv file..........')