# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import numpy as np
import os
from typing import List, Optional, Dict, Union
import pickle


class DielectricProperties:

    def __init__(self):
        # A list of strings containing the names of all the dielectrics
        self.dielectric_names = []
        """
        self.values is a dictionary containing one parameter for each key
        each key contains a numpy.ndarray (shape=(1,-1) => row vector), each value giving the parameter 
        for the dielectric at position i
        i.e.:
            self.values = {'param1': np.array([5, 4, 3], ndmin=2), 'param2': np.array([0, 1, 2], ndmin=2)}
        hence:
            param1 has the value 5 for dielectric at position 0 in self.dielectric_names
            param2 has the value 2 for dielectric at position 2 in self.dielectric_names
        """
        self.values = {}
        self.epsilon0 = 8.85418782e-12

    def load(self, filename: str):
        """
        Load the file containing the dict with the
        :return:
        """
        current_directory = os.path.dirname(__file__)
        filename = os.path.join(current_directory, filename)
        with open(filename, "rb") as f:
            data = pickle.load(f)

        # import the loaded data into the same instance of this class
        self.values.clear()
        self.values.update(data['values'])

        self.dielectric_names = data['dielectric_names']

    def save(self, filename):
        """
        Save the Dielectric properties parameter list

        :return:
        """
        dump_dict = {'dielectric_names': self.dielectric_names, 'values': self.values}

        current_directory = os.path.dirname(__file__)
        filename = os.path.join(current_directory, filename)
        with open(filename, "wb") as f:
            pickle.dump(dump_dict, f)

    def complex_permittivity(self, dielectric_index: np.ndarray, f: np.ndarray) -> np.ndarray:
        # this function has to be implemented in the child class
        # should return the complex permittivity for a given dielectric and frequency
        raise NotImplementedError

    def wave_impedance(self, dielectric_index: Union[np.ndarray, int], f: np.ndarray) -> np.ndarray:
        """
        Calculate the wave impedance of the tissues given in dielectric_index for frequencies f

        :param numpy.ndarray dielectric_index:
        :param numpy.ndarray f:
        :return: numpy.ndarray
        """
        if type(f) is not np.ndarray:
            # try to cast the input to a numpy array
            f = np.array([f])
        # make sure f is a column vector
        f.shape = (-1, 1)

        if type(dielectric_index) is not np.ndarray:
            # cast it to numpy array to fix the shape to (1, -1)
            dielectric_index = np.array(dielectric_index)
        # make sure that dielectric_index is a row vector
        dielectric_index.shape = (-1, )

        mu0 = 4 * np.pi * 1e-7
        epsilon = self.complex_permittivity(dielectric_index, f)
        eta = np.sqrt(mu0 / epsilon)

        return eta

    def propagation_constant(self, dielectric_index: np.ndarray, f: np.ndarray) -> np.ndarray:
        """
        Calculate the propagation constant for the tissues given in dielectric_index at frequencies f

        :param numpy.ndarray dielectric_index:
        :param numpy.ndarray f:
        :return: numpy.ndarray
        """
        if type(f) is not np.ndarray:
            # try to cast the input to a numpy array
            f = np.array([f])
        # make sure f is a column vector
        f.shape = (-1, 1)

        mu0 = 4 * np.pi * 1e-7
        epsilon = self.complex_permittivity(dielectric_index, f)
        # For the calculation f must be extended to a matrix of same dimension as epsilon
        f = np.tile(f, (1, len(dielectric_index)))
        gamma = 1j * 2 * np.pi * f * np.sqrt(mu0 * epsilon)

        return gamma

    def get_id_for_name(self, dielectric_name: Union[str, List[str]]) -> np.ndarray:
        """
        Returns a numpy array of the tissue indices given in the list of strings tissue_name

        :param [str] dielectric_name:
        :return: numpy.ndarray
        """
        # if its just one string, make one element list out of it
        if isinstance(dielectric_name, str):
            dielectric_name = [dielectric_name]

        # select only the rows from cole_cole_values that occur in tissue_type
        tissue_number = len(dielectric_name)
        tissue_index = np.zeros(tissue_number)

        for k in range(0, tissue_number):
            tissue_index[k] = self.dielectric_names.index(dielectric_name[k])

        return tissue_index.astype(int)

    def get_name_for_id(self, dielectric_index: Union[float, int, np.ndarray]) -> List[str]:
        """
        Resolves the tissue_id's given into the names from self.tissue_names

        :param numpy.ndarray dielectric_index:
        :return: [str]
        """
        # make sure dielectric_index is a numpy array
        if not isinstance(dielectric_index, np.ndarray):
            dielectric_index = np.array(dielectric_index)

        dielectric_names = []
        for k in dielectric_index:
            dielectric_names.append(self.dielectric_names[k])

        return dielectric_names

    def add_new_dielectric(self, dielectric_name: str, copy_values_from: Optional[str]=None,
                           new_values: Optional[Dict]=None):
        """
        Add a new tissue to the existing tissue data and automatically save the updated version of the file.

        :param dielectric_name:  name of the tissue that should be added to the tissues
        :param copy_values_from:    name of the tissue that the cole-cole-constants need be copied from. If empty
                                    the third parameter new_values is needed.
        :param new_values:          new_values contains the parameters for the specified tissue type for each
                                    parameter given in self.values
        """
        # append to the list of tissue names
        self.dielectric_names.append(dielectric_name)
        # copy the constants from another dielectric if desired
        if type(copy_values_from) is str:
            index = self.get_id_for_name(copy_values_from)
            for key, value in self.values.items():
                self.values[key] = np.hstack((value, value[:, index]))
        # otherwise append the new constants to the end of the values dict for each parameter
        else:
            if not self.values:
                # if self.values is empty use new_values as first entry
                self.values.update(new_values)
            else:
                # append the values given at the end of each array in self.values
                for key, value in self.values.items():
                    self.values[key] = np.hstack((value, new_values[key]))

