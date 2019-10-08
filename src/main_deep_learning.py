# Imported numpy
import numpy as np

def read_name_pdb_to_list(links, name_list):
    ''' Method which add all names of pdb into list. '''
    with open(links, "r") as list_pdb:
        for line in list_pdb:
            name_list.append(line.strip())

def create_dictonnary_of_matrix_npy(links, name_list_reduce):
    ''' Method which return dictionnary from lists of name pdb and
    associated the correct matrix .npy .'''
    basic_dict = dict()

    for i in range(0, len(name_list_reduce), 1):
        numpy_array = np.load(links+name_list_reduce[i]+".npy")
        control_dict = {name_list_reduce[i] : numpy_array}

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

    # Define the path of all numpy matrix.
    path_deepdrug3D = "../data/deepdrug3d_voxel_data/"

    # Create dictionnarys.
    control_dict = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                   control_list_reduce)

    heme_dict = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                heme_list_reduce)

    nucleotide_dict = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                      nucleotide_list_reduce)

    steroid_dict = create_dictonnary_of_matrix_npy(path_deepdrug3D,
                                                   steroid_list_reduce)
