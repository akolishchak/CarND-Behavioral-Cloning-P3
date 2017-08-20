#
#
#  CarND Behavior Cloning Project
#
#
import os
import cv2
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import Cropping2D, Lambda, Convolution2D, Activation, Flatten
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use('TkAgg')


def load_data(path, steering_correction=0.2):
    """
    Parse Udacity simulator data directory

    :param path:                path to data directory
    :param steering_correction: steering correction for left and right camera images

    :return: two lists - image file paths and corresponding steering angles
    """
    csv_file_name = path + 'driving_log.csv'
    if not os.path.exists(csv_file_name):
        raise Exception('File not found: ' + csv_file_name)

    images = []
    angles = []
    with open(csv_file_name, 'r') as f:
        path += 'IMG/'
        reader = csv.reader(f)
        is_header = True
        for row in reader:
            if is_header:
                is_header = False
                continue

            steering_center = float(row[3])
            #
            # create adjusted steering measurements for the side camera images
            #
            steering_left = steering_center + steering_correction
            steering_right = steering_center - steering_correction
            #
            # read in images from center, left and right cameras
            #
            img_center = path + row[0].split('/')[-1]
            img_left = path + row[1].split('/')[-1]
            img_right = path + row[2].split('/')[-1]
            #
            # add images and angles to dataset
            #
            images.extend([img_center, img_left, img_right])
            angles.extend([steering_center, steering_left, steering_right])

    return images, angles


def random_shear(image, steering, shear_range):
    """
    Random sheer of an image
     Source: https://github.com/ksakmann/CarND-BehavioralCloning

    :param image:       image array (height, width, channels)
    :param steering:    float steering angle
    :param shear_range: int sheer range in pixels

    :return: image array and adjusted steering angle
    """
    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)

    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering += dsteering

    return image, steering


def random_brightness(image):
    """
    Randomly adjust brightness of an image
     Source: https://github.com/ksakmann/CarND-BehavioralCloning

    :param image: image array (height, width, channels)

    :return: image array with randomly adjusted brightness
    """
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4 * (2 * np.random.uniform() - 1.0)
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


def random_flip(image, steering):
    """
    Random flipping of an image
    :param image:    image array (height, width, channels)
    :param steering: float steering angle

    :return: numpy array of image and adjusted steering angle
    """
    if np.random.randint(2) == 0:
        image = np.fliplr(image)
        steering = -steering

    return image, steering


def data_generator(image_paths, angles, batch_size):
    """
    Data generator that creates batches for keras.fit_generator
     It also performs data augmentation

    :param image_paths: list of image file names
    :param angles:      list of steering angles corresponding to image_paths
    :param batch_size:  size of batch

    :return: two numpy arrays - images and corresponding steering angles
    """
    num_samples = len(image_paths)
    while True:  # Loop forever so the generator never terminates
        shuffle(image_paths, angles)
        for offset in range(0, num_samples, batch_size):
            image_list = []
            angle_list = []
            for image_path, angle in zip(image_paths[offset:offset+batch_size], angles[offset:offset+batch_size]):
                #
                # load image from file
                #
                if not os.path.exists(image_path):
                    raise Exception('File not found: ' + image_path)
                image = plt.imread(image_path)
                #
                # augmentation
                #
                image, angle = random_shear(image, angle, shear_range=100)
                image, angle = random_flip(image, angle)
                image = random_brightness(image)

                image_list.append(image)
                angle_list.append(angle)

            yield sklearn.utils.shuffle(np.array(image_list), np.array(angle_list))


def create_model(input_shape, cropping):
    """
    Create Keras Model for steering angle policy

    :param input_shape: tuple of input dimensions (height, width, channels)
    :param cropping:    tuple of cropping borders for Cropping2D

    :return: Keras Model
    """
    model = Sequential()
    # cropping
    model.add(Cropping2D(cropping=cropping, input_shape=input_shape, name='Cropping'))

    # pre-processing
    model.add(Lambda(lambda x: x / 127.5 - 1., name='Normalization'))

    # convolution features
    model.add(Convolution2D(4, 3, 3, subsample=(2, 2), name='Convolution_4_3x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(8, 3, 3, subsample=(2, 2), name='Convolution_8_3x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(16, 3, 3, subsample=(2, 2), name='Convolution_16_3x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 3, 3, subsample=(2, 2), name='Convolution_32_3x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 2), name='Convolution_64_3x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 1, 3, subsample=(1, 2), name='Convolution_128_1x3'))
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 1, 3, subsample=(1, 2), name='Convolution_256_1x3'))
    model.add(Activation('relu'))

    # one to one convolutions for steering angle
    model.add(Convolution2D(64, 1, 1, subsample=(1, 1), name='Convolution_64_1x1'))
    model.add(Activation('relu'))

    model.add(Convolution2D(32, 1, 1, subsample=(1, 1), name='Convolution_32_1x1'))
    model.add(Activation('relu'))

    model.add(Convolution2D(1, 1, 1, subsample=(1, 1), name='Convolution_1_1x1'))

    # steering angle
    model.add(Flatten(name='steering'))

    return model


def train(args):
    """
    Main training procedure

    :param args: dictionary of parameters

    :return: None
    """
    #
    # load data
    #
    # track 1
    images, angles = load_data(args.data_path + 'track1_1/', steering_correction=0.12)
    # track 2
    track2_images, track2_angles = load_data(args.data_path + 'track2_1/', steering_correction=0.2)
    images.extend(track2_images)
    angles.extend(track2_angles)
    track2_images, track2_angles = load_data(args.data_path + 'track2_2/', steering_correction=0.2)
    images.extend(track2_images)
    angles.extend(track2_angles)
    track2_images, track2_angles = load_data(args.data_path + 'track2_3/', steering_correction=0.2)
    images.extend(track2_images)
    angles.extend(track2_angles)
    #
    # split into training and validation sets
    #
    train_images, val_images, train_angles, val_angles = train_test_split(images, angles, test_size=0.1)
    #
    # create data generators
    #
    train_generator = data_generator(train_images, train_angles, args.batch_size)
    validation_generator = data_generator(val_images, val_angles, args.batch_size)
    #
    # create model
    #
    input_shape = (160, 320, 3)
    model = create_model(input_shape=input_shape, cropping=((70, 25), (0, 0)))
    model.summary()
    #
    # train the model
    #
    model.compile(loss='mse', optimizer='adam')
    history = model.fit_generator(train_generator, samples_per_epoch=len(train_images),
                                  validation_data=validation_generator, nb_val_samples=len(val_images),
                                  nb_epoch=args.epoch_num)
    #
    # save the trained model
    #
    model.save(args.model_file_name)
    #
    # print training results
    #
    print(history)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CarND Behavior Cloning')
    parser.add_argument('--data_path', default=os.path.expanduser('~') + '/test/datasets/udacity_car_simulator/',
                        help='path to udacity car simulator data')
    parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
    parser.add_argument('--epoch_num', type=int, default=30, help='number of epochs per training')
    parser.add_argument('--model_file_name', default='model.h5', help='file name of the model')

    args = parser.parse_args()

    train(args)
