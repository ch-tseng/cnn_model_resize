from keras.datasets import cifar10
from keras.layers import Activation
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.datasets import mnist
from keras.utils import np_utils  
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
from keras.utils import to_categorical
import matplotlib.pyplot as plt
%matplotlib inline

model_type = "custom_cnn"
# training parameters
batch_size = 32  # orig paper trained all networks with batch_size=128
epochs = 50
num_classes = 10
data_augmentation = True
num_classes = 10
# subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions.
input_shape = x_train.shape[1:]

# normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# if subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# convert class vectors to binary class matrices.
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 e$
    Called automatically every epoch as part of callbacks during train$

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def custom_cnn(model):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, kernel_size=(5,5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    return model

def separable_cnn(model):

    model.add(SeparableConv2D(32, (5,5), input_shape=input_shape, activation='relu', name='SeparableConv2D-1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(16, (5,5), activation='relu', name='SeparableConv2D-2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten()) 
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    return model

def gap_cnn(model):

    model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape, activation='relu'))
    model.add(Conv2D(16, kernel_size=(5,5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    gap = GlobalAveragePooling2D()(base_model)
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    return model

def gap_separable_cnn(model):
    
    model.add(SeparableConv2D(32, (5,5), input_shape=input_shape, activation='relu', name='SeparableConv2D-1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SeparableConv2D(16, (5,5), activation='relu', name='SeparableConv2D-2'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    gap = GlobalAveragePooling2D()(base_model)
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.summary()
    return model

model = Sequential()
model = custom_cnn(model)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])

# prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # this will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test), steps_per_epoch=x_train.shape[0],
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

#score trained model
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
