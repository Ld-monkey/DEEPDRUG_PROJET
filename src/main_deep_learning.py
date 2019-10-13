# Imported numpy
import numpy as np

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
    control_list_reduce = np.random.choice(control_list,
                                           int(len(control_list)/10),
                                           replace = False)

    heme_list_reduce = np.random.choice(heme_list,
                                        int(len(heme_list)/10),
                                        replace = False)
    nucleotide_list_reduce = np.random.choice(nucleotide_list,
                                              int(len(nucleotide_list)/10),
                                              replace = False)

    steroid_list_reduce = np.random.choice(steroid_list,
                                           int(len(steroid_list)/10),
                                            replace = False)

    # A complet list is result of all concatenations of all previous datas.
    complet_list_all_data = np.concatenate((control_list_reduce,
                                            heme_list_reduce,
                                            nucleotide_list_reduce))
    # print(complet_list_all_data)


    # Define the path of all numpy matrix.
    path_deepdrug3D = "../data/deepdrug3d_voxel_data/"

    # Create Y_train
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
    print(y_train)

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

    # Create model
