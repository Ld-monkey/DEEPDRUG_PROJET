import numpy as np
import matplotlib.pyplot as plt
import keras

from keras import Input, Model, Sequential
from keras.layers import Convolution3D, AveragePooling1D
from keras.layers import Activation, Dropout, Flatten
from keras.layers import Dense, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as t


def read_name_pdb_to_list(links, name_list):
    ''' Method which add all names of pdb into list. '''
    with open(links, "r") as list_pdb:
        for line in list_pdb:
            name_list.append(line.strip())

def create_dictonnary_of_matrix_npy(links, name_list_reduce):
    ''' Method which return a list from lists  correct matrix .npy .'''
    basic_dict = list()

    for i in range(0, len(name_list_reduce), 1):
        numpy_array = np.load(links+name_list_reduce[i]+".npy")
        basic_dict.append(numpy_array)

    print("Finish !")
    return basic_dict

if __name__ == "__main__":
    print("DEEP DRUG 3D - 2019 - 2020")

    # Define all lists
    control_list = list()
    heme_list = list()
    nucleotide_list = list()
    steroid_list = list()

    # Add control list.
    read_name_pdb_to_list("../data/control_list/control.list", control_list)

    # Add heme list.
    read_name_pdb_to_list("../data/heme_list/heme.list", heme_list)

    # Add nucleotide list.
    read_name_pdb_to_list("../data/nucleotide_list/nucleotide.list", nucleotide_list)

    # Add steroid list.
    read_name_pdb_to_list("../data/steroid_list/steroid.list", steroid_list)

    # 1/10 of lists to reduce data.

    REDUCE_VARIABLE = 50

    control_list_reduce = np.random.choice(control_list,
                                           int(len(control_list)/REDUCE_VARIABLE),
                                           replace = False)

    heme_list_reduce = np.random.choice(heme_list,
                                        int(len(heme_list)/REDUCE_VARIABLE),
                                        replace = False)
    nucleotide_list_reduce = np.random.choice(nucleotide_list,
                                              int(len(nucleotide_list)/REDUCE_VARIABLE),
                                              replace = False)

    steroid_list_reduce = np.random.choice(steroid_list,
                                           int(len(steroid_list)/REDUCE_VARIABLE),
                                            replace = False)

    # A complet list is result of all concatenations of all previous datas.
    complet_list_all_data = np.concatenate((control_list_reduce,
                                            heme_list_reduce,
                                            nucleotide_list_reduce))
    # print(complet_list_all_data)


    # Define the path of all numpy matrix.
    path_deepdrug3D = "../data/deepdrug3d_voxel_data/"

    # Define y_train
    """
    a = np.array([1, 0, 0]* len(control_list_reduce) * 14 * 32 * 32 * 32).reshape((len(control_list_reduce) * 14* 32* 32* 32, 3))
    print(a.shape)
    """
    y_train = list()

    for i in range(0, len(control_list_reduce)):
        if i <= len(control_list_reduce):
            y_train.append([1,0,0])

    for i in range(0, len(heme_list_reduce)):
        if i <= len(heme_list_reduce):
            y_train.append([0,1,0])

    for i in range(0, len(nucleotide_list_reduce)):
        if i <= len(nucleotide_list_reduce):
            y_train.append([0,0,1])

    all_elements_reduce = len(control_list_reduce) + len(heme_list_reduce) + len(nucleotide_list_reduce)

    y_train = np.reshape(a = y_train, newshape = (all_elements_reduce, 3))
    print(y_train.shape)

    # Create x_control train.
    control_list = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                   control_list_reduce)

    # Visualize voxel data.
    control_list_np = np.array(control_list)

    # Reduce dimention au control data set.
    control_list_np = np.squeeze(control_list_np, axis = 1)
    print(control_list_np.shape)

    # Create nucleatide list for x_train.
    nucleotide_train = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                       nucleotide_list_reduce)

    nucleotide_list_np = np.array(nucleotide_train)
    # Display dimension of nucleotide train
    print(nucleotide_list_np.shape)

    # Create heme list for x_train.
    heme_train = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                 heme_list_reduce)

    heme_list_np = np.array(heme_train)
    # Display dimension of heme train
    print(heme_list_np.shape)

    # Define x_train with all datas.
    x_train = np.concatenate((control_list_np,
                              nucleotide_list_np,
                              heme_list_np))
    print(x_train.shape)

    # Create deep learning model.
    print("Model of deep learning")

    """
    model = Sequential()

    #inputs = Input(shape=(14, 32, 32, 32))

    # Two convolutional layers with leaky Relu activations functions.
    model.add(Convolution3D(filters = 64,
                            kernel_size = 5,
                            padding = 'valid',
                            activation = 'relu',
                            input_shape = (14, 32, 32, 32)))

    model.add(Convolution3D(filters = 64,
                            kernel_size = 3,
                            padding = 'valid',
                            activation = 'relu',
                            data_format='channels_first'))

    # A serie of dropout, pooling fully connected and softmax layers.
    model.add(Dropout(0.2))
    model.add(MaxPooling3D(pool_size =(2, 2, 2),
                           strides = None,
                           padding = 'valid',
                           data_format = 'channels_first'))
    model.add(Dropout(0.4))
    # Fully connected 512
    model.add(Dropout(0.4))
    model.add(Activation('softmax'))

    # Compiling model.
    model.compile(optimizer = "adam", loss="categorical_crossentropy", metrics=["accuracy"])
    """

    model = Sequential()

    # Conv layer 1
    model.add(Convolution3D(
        input_shape = (14,32,32,32),
        filters=64,
        kernel_size=5,
        padding='valid',     # Padding method
        data_format='channels_first',
        ))

    model.add(LeakyReLU(alpha = 0.1))
    # Dropout 1
    model.add(Dropout(0.2))
    # Conv layer 2
    model.add(Convolution3D(
        filters=64,
        kernel_size=3,
        padding='valid',     # Padding method
        data_format='channels_first',
    ))

    model.add(LeakyReLU(alpha = 0.1))
    # Maxpooling 1
    model.add(MaxPooling3D(
        pool_size=(2,2,2),
        strides=None,
        padding='valid',    # Padding method
        data_format='channels_first'
    ))

    # Dropout 2
    model.add(Dropout(0.4))

    # FC 1
    model.add(Flatten())
    model.add(Dense(128)) # TODO changed to 64 for the CAM
    model.add(LeakyReLU(alpha = 0.1))

    # Dropout 3
    model.add(Dropout(0.4))

    # Fully connected layer 2 to shape (2) for 2 classes
    model.add(Dense(3))
    model.add(Activation('softmax'))

     #adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    adam = Adam(learning_rate = 0.00001, beta_1 = 0.9,
                beta_2 = 0.999, amsgrad = False, decay = 0.0)

    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])


    # Save history of model. Ne pas oublier de mettre les tests de validations
    print(y_train.shape)
    print(x_train.shape)
    history = model.fit(x_train, y_train, epochs = 50, batch_size = 16, validation_split = 0.1)

    """
    # Evaluate model.
    score = model.evaluate(np.array(x_test),
                           np.array(y_test),
                           verbose = 1)

    print("Test score : ", score[0])
    print("Test accuracy : ", score[1])

    """
    print(history.history.keys())

    # summarize history for accuracy.
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss.
    plt.plot(history.history['loss'])
    plt.plot(history.history["val_loss"])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
