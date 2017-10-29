import numpy as np
import cv2
import pandas as pd
import pickle
import matplotlib.image as mpimg
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Model
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('log_file', 'data/driving_log.csv', "Driving log file (.csv).")
flags.DEFINE_string('image_path', 'data/IMG/', "Image path.")
flags.DEFINE_string('output_name', 'model', "Output name for Keras model file (.h5) and training log (.p).")
flags.DEFINE_float('test_size', 0.1, "Validation sample proportion.")
flags.DEFINE_float('dropout_rate', 0.2, "Dropout rate for fc layers.")
flags.DEFINE_integer('epochs', 10, "The number of epochs.")
flags.DEFINE_integer('batch_size', 32, "The batch size.")


# preprocessing parameters
correction = 0.5
top_crop, bottom_crop = 50, 20
left_crop, right_crop = 0, 0

def generator(samples, batch_size):
    num_samples = len(samples)
    batch_size_new = int(np.floor(batch_size/2))
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size_new):
            batch_samples = samples[offset:offset+batch_size_new]

            images = []
            angles = []
            for batch_sample in batch_samples:
                corrections=[0,correction,-correction]
                for i in range(3):
                    source_path = batch_sample[i]
                    filename = source_path.split('/')[-1]
                    name = FLAGS.image_path+filename
                    center_image = mpimg.imread(name)
                    center_angle = float(batch_sample[3])+corrections[i]
                    images.append(center_image)
                    angles.append(center_angle)
                    images.append(cv2.flip(center_image,1))
                    angles.append(center_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)


def main(_):
    # load driving log
    log = pd.read_csv(FLAGS.log_file)

    samples = []
    for row in log.iterrows():
        index, data = row
        samples.append(data.tolist())
    
    # train validation data split
    train_samples, validation_samples = train_test_split(samples, test_size=FLAGS.test_size)

    # model parameters
    learning_rate = 0.001

    # define model
    input_shape = (160,320,3)
    inp = Input(shape=input_shape)
    x = Cropping2D(cropping=((top_crop, bottom_crop), (left_crop, right_crop)))(inp)
    x = Lambda(lambda x: (x / 255.0) - 0.5)(x)
    x = Convolution2D(12,5,5,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Convolution2D(24,5,5,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)
    x = Convolution2D(48,3,3,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(48,3,3,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(100,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(FLAGS.dropout_rate)(x)
    x = Dense(50,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(FLAGS.dropout_rate)(x)
    x = Dense(10,activation='elu',bias=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(FLAGS.dropout_rate)(x)
    x = Dense(1,activation='tanh')(x)
    model = Model(inp, x)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    train_generator = generator(train_samples, batch_size=FLAGS.batch_size)
    validation_generator = generator(validation_samples, batch_size=FLAGS.batch_size)

    # train model
    history_object = model.fit_generator(train_generator, 
                                         samples_per_epoch= len(train_samples)*6, 
                                         validation_data=validation_generator, 
                                         nb_val_samples=len(validation_samples)*6, 
                                         nb_epoch=FLAGS.epochs,
                                         verbose=1)
    train_log = history_object.history
    model.save(FLAGS.output_name+'.h5')
    with open(FLAGS.output_name+'.p',mode='wb') as f:
        pickle.dump(train_log, f)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
