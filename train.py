from datetime import datetime
from dataset import Dataset
from model import CNN2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.optimizers import SGD
from keras import optimizers


def sec2time(elapsed_sec):
    hours, remaining = divmod(elapsed_sec, 3600)
    minutes, seconds = divmod(remaining, 60)

    return '{:>02} hours {:>02} minutes {:>05.2f} seconds.'.format(int(hours), int(minutes), int(seconds))


def train(n_epochs, batch_size, image_shape):
    # helpers
    # 학습이 정체되면 학습속도를 줄인다.
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=10, min_lr=0.000001, verbose=1)
    # checkpointer
    checkpointer = ModelCheckpoint(filepath='./log/train/checkpoint.hdf5', verbose=0, save_best_only=False)
    # tensorboard
    tensorboard  = TensorBoard(log_dir='./log/train/', write_graph=True)
    # earlystopper
    earlystopper = EarlyStopping(patience=10)
    # csv logger
    csvlogger    = CSVLogger('./log/train/learning.log')

    # create dataset
    data = Dataset(batch_size, image_shape)

    train_steps_per_epoch = len(data.train) // batch_size
    validation_steps_per_epoch = len(data.test) // batch_size

    # create data generator
    train_generator = data.image_generator('train')
    validation_generator = data.image_generator('test')
    
    # create model
    model = CNN2D(len(data.classes), image_shape)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=0.0002),
              metrics=['accuracy'])

    model.summary()

    # training
    history = model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        callbacks=[reduce_lr, checkpointer, tensorboard, earlystopper, csvlogger],
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=3
    )


    # save weights
    model.save_weights('./log/train/weights.h5')
    model.save('./log/train/model_epochs_' + str(n_epochs) + '.h5')
    

def main():
    # hyper parameters
    n_epochs = 100
    batch_size = 32
    image_shape = (224, 224, 3)

    # start time
    start = datetime.now()

    # learning
    train(n_epochs, batch_size, image_shape)

    # end time
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print('\n{:>10}  {}\n'.format('[ELAPSED]', sec2time(elapsed)))


if __name__ == '__main__':
    main()