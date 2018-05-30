from datetime import datetime
from dataset import Dataset
from keras.models import load_model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras.optimizers import SGD


def sec2time(elapsed_sec):
    hours, remaining = divmod(elapsed_sec, 3600)
    minutes, seconds = divmod(remaining, 60)

    return '{:>02} hours {:>02} minutes {:>05.2f} seconds.'.format(int(hours), int(minutes), int(seconds))


def retrain(n_epochs, batch_size, image_shape, saved_model):
    # helpers
    checkpointer = ModelCheckpoint(filepath='./log/retrain/checkpoint.hdf5', verbose=0, save_best_only=True)
    tensorboard  = TensorBoard(log_dir='./log/retrain/', write_graph=False)
    csvlogger    = CSVLogger('./log/retrain/learning.log')

    # create dataset
    data = Dataset(batch_size, image_shape)

    train_steps_per_epoch = len(data.train) // batch_size
    validation_steps_per_epoch = len(data.test) // batch_size

    # create data generator
    train_generator = data.image_generator('train')
    validation_generator = data.image_generator('test')

    # create model
    model = load_model(saved_model)

    # enable to train all layers
    for layer in model.layers:
        layer.trainable = True

    # recompile for training
    optimizer = SGD(lr=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.summary()

    # training
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=n_epochs,
        verbose=1,
        callbacks=[checkpointer, tensorboard, csvlogger],
        validation_data=validation_generator,
        validation_steps=validation_steps_per_epoch,
        max_queue_size=3
    )

    # save weights
    model.save_weights('./log/retrain/weights.hdf5')


def main():
    # hyper parameters
    n_epochs = 100
    batch_size = 128
    image_shape = (80, 80, 3)
    saved_model = './log/train/checkpoint.hdf5'

    # start time
    start = datetime.now()

    # learning
    retrain(n_epochs, batch_size, image_shape, saved_model)

    # end time
    end = datetime.now()
    elapsed = (end - start).total_seconds()

    print('\n{:>10}  {}\n'.format('[ELAPSED]', sec2time(elapsed)))


if __name__ == '__main__':
    main()