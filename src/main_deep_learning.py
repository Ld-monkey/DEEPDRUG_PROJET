# coding : utf-8

"""
@author : Zygnematophyce
Master II BIB - 2019 2020
Projet Deep Learning
"""

# All imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as t
import keras

from keras import Input, Model, Sequential
from keras.layers import Convolution3D, AveragePooling1D
from keras.layers import Activation, Dropout, Flatten
from keras.layers import Dense, MaxPooling3D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam

# Allow to ignore tensorflow warning.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

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

    # Reduce variable 
    #REDUCE_VARIABLE = 10

    # Prendre la meme quantitée de poches. 100, 100, 100
    # Suffle dans le meme sens le x_train et y_train.

    control_list_reduce = np.random.choice(control_list,
                                           100,
                                           replace = False)

    heme_list_reduce = np.random.choice(heme_list,
                                        100,
                                        replace = False)
    nucleotide_list_reduce = np.random.choice(nucleotide_list,
                                              100,
                                              replace = False)

    steroid_list_reduce = np.random.choice(steroid_list,
                                           1,
                                            replace = False)

    # A complet list is result of all concatenations of all previous datas.
    complet_list_all_data = np.concatenate((control_list_reduce,
                                            nucleotide_list_reduce,
                                            heme_list_reduce))
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

    for i in range(0, len(nucleotide_list_reduce)):
        if i <= len(nucleotide_list_reduce):
            y_train.append([0,1,0])

    for i in range(0, len(heme_list_reduce)):
        if i <= len(heme_list_reduce):
            y_train.append([0,0,1])

    all_elements_reduce = len(control_list_reduce) + len(nucleotide_list_reduce) + len(heme_list_reduce)

    y_train = np.reshape(a = y_train, newshape = (all_elements_reduce, 3))
    print(y_train.shape)

    # Create x_control train.
    control_list_x_train = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                           control_list_reduce)

    # Visualize voxel data.
    control_list_np = np.array(control_list_x_train)

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

    # Suffle the two list in same order.

    print("Before suffleling.")

    #print(x_train.shape[0])
    #print(y_train)
    indice = np.arange(x_train.shape[0])
    np.random.shuffle(indice)

    x_train = x_train[indice]
    y_train = y_train[indice]

    print("-------- Shuffle x_train and y_train. actived ------------")

    # Define x_test and y_test

    print("-------- Start x_test and y_test.  -----------------------")
    """
    x_test = hazard(50 controles + 50 nucléotides + 50 hemes )
    y_test = 50x[1,0,0]+50x[0,1,0]+50x[0,0,1]
    --> suffle dans le meme ordre
    """

    x_test_control_list_reduce = np.random.choice(control_list,
                                                  50,
                                                  replace = False)

    x_test_heme_list_reduce = np.random.choice(heme_list,
                                               50,
                                               replace = False)
    x_test_nucleotide_list_reduce = np.random.choice(nucleotide_list,
                                                     50,
                                                     replace = False)

    x_test_complet_list_all_data = np.concatenate((x_test_control_list_reduce,
                                                   x_test_nucleotide_list_reduce,
                                                   x_test_heme_list_reduce))

    y_test = list()

    for i in range(0, len(x_test_control_list_reduce)):
        if i <= len(x_test_control_list_reduce):
            y_test.append([1,0,0])

    for i in range(0, len(x_test_nucleotide_list_reduce)):
        if i <= len(x_test_nucleotide_list_reduce):
            y_test.append([0,1,0])

    for i in range(0, len(x_test_heme_list_reduce)):
        if i <= len(x_test_heme_list_reduce):
            y_test.append([0,0,1])

    y_test_all_elements_reduce = len(x_test_control_list_reduce) + len(x_test_nucleotide_list_reduce) + len(x_test_heme_list_reduce)

    y_test = np.reshape(a = y_test, newshape = (y_test_all_elements_reduce, 3))
    print(y_train.shape)

     # Create x_control test.
    x_test_control_list = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                          x_test_control_list_reduce)

    x_test_control_list_np = np.array(x_test_control_list)

    # Reduce dimention au control data set.
    x_test_control_list_np = np.squeeze(x_test_control_list_np, axis = 1)
    print(x_test_control_list_np.shape)

    # Create nucleatide list for x_test.
    x_test_nucleotide_train = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                              x_test_nucleotide_list_reduce)

    x_test_nucleotide_list_np = np.array(x_test_nucleotide_train)

    # Display dimension of nucleotide train
    print(x_test_nucleotide_list_np.shape)

    # Create heme list for x_train.
    x_test_heme = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                        x_test_heme_list_reduce)

    x_test_heme_list_np = np.array(x_test_heme)

    # Display dimension of heme test
    print(x_test_heme_list_np.shape)

    # Define x_train with all datas.
    x_test = np.concatenate((x_test_control_list_np,
                             x_test_nucleotide_list_np,
                             x_test_heme_list_np))
    print(x_test.shape)

    # Suffle the two list in same order.
    print("Before suffleling.")
    indice_test = np.arange(x_test.shape[0])
    np.random.shuffle(indice_test)

    x_test = x_test[indice_test]
    y_test = y_test[indice_test]

    print("-------- Shuffle x_test and y_test. actived ------------")

    # Create deep learning model.
    print("Model of deep learning")

    # input layer
    inputs_layer = Input(shape=(14, 32, 32, 32))

    # convolutional layers
    conv_layer1 = Convolution3D(filters = 64,
                                kernel_size = (5, 5, 5),
                                padding='valid',
                                activation = "relu",
                                data_format ='channels_first')(inputs_layer)

    conv_layer2 = Convolution3D(filters = 64,
                                padding = 'valid',
                                kernel_size = (3, 3, 3),
                                activation = "relu",
                                data_format ='channels_first')(conv_layer1)

    # Dropout = 0.2
    dropout_layer2 = Dropout(0.2)(conv_layer2)

    # MaxPooling3D = 2
    pooling_drop2 = MaxPooling3D(pool_size = (2, 2, 2),
                                 data_format ='channels_first',
                                 padding = 'same')(dropout_layer2)

    # Dropout = 0.4
    dropout_pooling2 = Dropout(0.4)(pooling_drop2)

    # Flatten
    flatten_layer = Flatten()(dropout_pooling2)

    # Fully connected = 512
    dense_drop2 = Dense(512, activation='relu')(flatten_layer)

    # Dropout = 0.4
    dropout_dense2 = Dropout(0.4)(dense_drop2)

    # Output
    output_layer2 = Dense(units = 3, activation = "softmax")(dropout_dense2)

    # Define model with input layer and output layer.
    model = Model(inputs = inputs_layer, outputs = output_layer2)

    #model summarize
    model.summary()

    # Adam parameter.
    adam = Adam(learning_rate = 0.00001, beta_1 = 0.9,
                beta_2 = 0.999, amsgrad = False, decay = 0.0)

    # Compiling model.
    model.compile(optimizer = adam, loss="categorical_crossentropy", metrics=["accuracy"])

    # Save history of model. Ne pas oublier de mettre les tests de validations
    print("y_train : {}".format(y_train.shape))
    print("x_train : {}".format(x_train.shape))

    # epochs = 30 and batch_size = 32 // problème de dimension.
    history = model.fit(x = x_train,
                        y = y_train,
                        epochs = 30,
                        batch_size = 32,
                        validation_data = (x_test, y_test))


    # Evaluate model.
    score = model.evaluate(x_test,
                           y_test,
                           verbose=1)

    print("Test score : ", score[0])
    print("Test accuracy : ", score[1])


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

    # Save model.
    model.save('..result/model/deep_learning_model.h5')

    # model = load_model("my_model.h5")

    """
    Faire un cross-validation
    +
    courbe roc
    """

    """
    Dernier recourd essayer avec d'autre modèle
    et peut etre les comparer.
    """
