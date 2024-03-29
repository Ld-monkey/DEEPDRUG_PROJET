# coding : utf-8

"""
@author : Zygnematophyce
Master II BIB - 2019 2020
Projet Deep Learning
"""

# All imports
import os
import math
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
from keras.models import load_model

# Using for roc curve.
from sklearn import metrics

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

def general_statistic_calcul(true_positive, true_negative,
                             false_positive, false_negative):
    """ Methode calcul general statistic values from model. """
    acc = (true_positive + true_negative)/(true_positive + false_positive + true_negative + false_negative)

    ppv = (true_positive)/(true_positive + false_positive)

    tnr = (true_negative)/(true_negative + false_positive)

    tpr = (true_positive)/(true_positive + false_negative)

    fpr = (false_positive)/(false_positive + true_negative)

    mcc = ((true_positive * true_positive)-(false_positive * false_negative))/math.sqrt((true_positive + false_negative)*(true_positive + false_negative)*(true_negative + false_positive)*(true_negative + false_negative))
    print("ACC = {}\nPPV = {}\nTNR = {}\nTPR = {}\nFRP = {}\nMCC = {}".format(acc, ppv, tnr, tpr, fpr, mcc))

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

    #  Add seed to control randomization.
    np.random.seed(seed = 42)

    # Create under datasets
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

    # Define the path of all numpy matrix.
    path_deepdrug3D = "../data/deepdrug3d_voxel_data/"

    # Define y_train
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

    # Suffle the two list in same order.
    indice = np.arange(x_train.shape[0])
    np.random.shuffle(indice)

    x_train = x_train[indice]
    y_train = y_train[indice]

    print("-------- Shuffle x_train and y_train. actived ------------")

    # Define x_test and y_test

    print("-------- Start x_test and y_test.  -----------------------")
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

    # Define y test.
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

    # Create heme list for x_train.
    x_test_heme = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                        x_test_heme_list_reduce)

    x_test_heme_list_np = np.array(x_test_heme)

    # Define x_train with all datas.
    x_test = np.concatenate((x_test_control_list_np,
                             x_test_nucleotide_list_np,
                             x_test_heme_list_np))

    # Suffle the two list in same order.
    indice_test = np.arange(x_test.shape[0])
    np.random.shuffle(indice_test)

    x_test = x_test[indice_test]
    y_test = y_test[indice_test]

    print("-------- Shuffle x_test and y_test. actived ------------")

    verify_exit_model = os.path.exists('../results/model/deep_learning_model.h5')

    # Condition ask to delete deep learning model.
    if (verify_exit_model == True):
        print("Verify model.")
        print(verify_exit_model)
        reponse = False
        while( reponse != True):
            print("Do you want to remove deep_learning_model.h5 ?")
            delete_model = input("Please enter yes or no :")
            if (delete_model == 'yes'):
                os.remove('../results/model/deep_learning_model.h5')
                print('File deleted')
                reponse = True
            elif (delete_model == 'no'):
                print('File no deleted')
                reponse = True
            else:
                reponse = False

    if (verify_exit_model != True):
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

        # epochs = 30 and batch_size = 32 // problème de dimension.
        history = model.fit(x = x_train,
                            y = y_train,
                            epochs = 30,
                            batch_size = 32,
                            validation_data = (x_test, y_test))

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
        model.save('../results/model/deep_learning_model.h5')
    else:
        print("../results/model/deep_learning_model.h5 exits")
        model = load_model("../results/model/deep_learning_model.h5")

    # Evaluate model.
    score = model.evaluate(x_test,
                           y_test,
                           verbose=1)

    print("Test score : ", score[0])
    print("Test accuracy : ", score[1])
    print("-----------------")

    all_x_test_string = np.concatenate((x_test_control_list_reduce,
                                       x_test_nucleotide_list_reduce,
                                       x_test_heme_list_reduce))
    print("----------------")

    # Define Y test prediction.
    y_test_prediction = model.predict(x_test)

    # For control prediction.
    y_test_prediction_control = y_test_prediction[:,0]
    y_test_control = y_test[:,0]

    print(y_test_control)
    print(y_test_prediction_control.round())

    """
    This part of the code highlights the false positives to try to
    understand why they are interpreted as false positive
    """
    FN = 0
    FP = 0
    TP = 0
    TN = 0

    for i in range(len(y_test_prediction_control)): 
        if y_test_prediction_control.round()[i]==1 and y_test_control[i]!=y_test_prediction_control[i].round():
            print("False Positive")
            FP += 1
            print(i)
            print(indice_test[i])
            print(all_x_test_string[indice_test[i]])
            print(y_test[i])
        elif y_test_prediction_control.round()[i]==0 and y_test_control[i]!=y_test_prediction_control[i].round():
           print("False Negative")
           FN += 1
           print(i)
           print(indice_test[i])
           print(all_x_test_string[indice_test[i]])
           print(y_test[i])
        if y_test_control[i]==y_test_prediction_control.round()[i]==1:
           TP += 1
        if y_test_control[i]==y_test_prediction_control.round()[i]==0:
           TN += 1

    print("FN = ", FN)
    print("FP = ", FP)
    print("VP = ", TP)
    print("FN = ", TN)

    exit()

    control_fpr, control_tpr, control_thresholds = metrics.roc_curve(y_test_control, y_test_prediction_control)
    control_roc_auc = metrics.auc(control_fpr, control_tpr)

    # For nucleotide prediction.
    y_test_prediction_nucleotide = y_test_prediction[:,1]
    y_test_nucleotide = y_test[:,1]

    nucleotide_fpr, nucleotide_tpr, nucleotide_thresholds = metrics.roc_curve(y_test_nucleotide, y_test_prediction_nucleotide)
    nucleotide_roc_auc = metrics.auc(nucleotide_fpr, nucleotide_tpr)

    # For heme prediction.
    y_test_prediction_heme = y_test_prediction[:,2]
    y_test_heme = y_test[:,2]

    heme_fpr, heme_tpr, heme_thresholds = metrics.roc_curve(y_test_heme, y_test_prediction_heme)
    heme_roc_auc = metrics.auc(heme_fpr, heme_tpr)

    # Plot of roc curve.
    plt.title("Courbe ROC")
    plt.plot(control_fpr,
             control_tpr,
             color='darkblue',
             linestyle='-',
             linewidth=4,
             label = "control AUC = %0.2f" % control_roc_auc)

    plt.plot(heme_fpr,
             heme_tpr,
             color='darkorange',
             linestyle=':',
             linewidth=4,
             label = "heme AUC = %0.2f" % heme_roc_auc)

    plt.plot(nucleotide_fpr,
             nucleotide_tpr,
             color='deeppink',
             linestyle=':',
             linewidth=4,
             label = "nucleotide AUC = %0.2f" % nucleotide_roc_auc)

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel("Sensibility")
    plt.xlabel("1-Sensibility")
    plt.show()

    # for confusion matrix of control.
    tn, fp, fn , tp = metrics.confusion_matrix(y_test_control,
                                               y_test_prediction_control.round()).ravel()
    print("For control\nVP = {} | FP = {} \nFN = {} | VN = {}".format(tp, fp, fn, tn))

    # 

    # Display statistics values
    general_statistic_calcul(tp, tn, fp, fn)

    # for confusion matrix of nucleotide.
    tn, fp, fn , tp = metrics.confusion_matrix(y_test_nucleotide,
                                               y_test_prediction_nucleotide.round()).ravel()
    print("For nucleotide\nVP = {} | FP = {} \nFN = {} | VN = {}".format(tp, fp, fn, tn))

    # Display statistics values
    general_statistic_calcul(tp, tn, fp, fn)

    # for confusion matrix of heme.
    tn, fp, fn , tp = metrics.confusion_matrix(y_test_heme,
                                               y_test_prediction_heme.round()).ravel()
    print("For heme\nVP = {} | FP = {} \nFN = {} | VN = {}".format(tp, fp, fn, tn))

    # Display statistics values
    general_statistic_calcul(tp, tn, fp, fn)
