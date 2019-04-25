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
from typing import List, Optional, Union, Dict

from LayerModel_lib.dielectric import DielectricProperties


class TissueProperties(DielectricProperties):
    """"
    A class for the Tissue Properties and their dielectric properties
    """

    def __init__(self):
        DielectricProperties.__init__(self)

        self.load('ColeColeConstants.TissueProperties')
        
        # tissue names refers to the same list as dielectric names
        # master !
        self.tissue_names = self.dielectric_names

    def complex_permittivity(self, tissue_index: Union[np.ndarray, float],
                             f: Union[np.ndarray, float]) -> np.ndarray:
        """"
        Returns the complex permittivity of tissue_index at frequency f. Based on the Cole-Cole equations and
        the data from Gabriel et. al

        :param numpy.ndarray tissue_index: array containing the indices to the tissue names in self.tissue_names
        :param numpy.ndarray f:  the frequency in Hz (column vector)
        :return:
        numpy.ndarray epsilon   the complex permittivity
                                for each frequency one row and
                                for each tissue_type one column

        """
        if type(f) is not np.ndarray:
            # try to cast the input to a numpy array
            f = np.array([f])
        # make sure f is a column vector
        f.shape = (-1, 1)

        if type(tissue_index) is not np.ndarray:
            # try to cast the input to a numpy array
            tissue_index = np.array([tissue_index])

        omega = 2 * np.pi * f

        tissue_number = len(tissue_index)

        # Load the constants from the dict self.values into the according variables
        delta = np.zeros((4, tissue_number))
        alpha = np.zeros((4, tissue_number))
        tau = np.zeros((4, tissue_number))
        
        ef = self.values['ef'][:, tissue_index]
        delta[0, :] = self.values['del1'][:, tissue_index]
        tau[0, :] = self.values['tau1'][:, tissue_index]
        alpha[0, :] = self.values['alf1'][:, tissue_index]
        delta[1, :] = self.values['del2'][:, tissue_index]
        tau[1, :] = self.values['tau2'][:, tissue_index]
        alpha[1, :] = self.values['alf2'][:, tissue_index]
        sigma = self.values['sig'][:, tissue_index]
        delta[2, :] = self.values['del3'][:, tissue_index]
        tau[2, :] = self.values['tau3'][:, tissue_index]
        alpha[2, :] = self.values['alf3'][:, tissue_index]
        delta[3, :] = self.values['del4'][:, tissue_index]
        tau[3, :] = self.values['tau4'][:, tissue_index]
        alpha[3, :] = self.values['alf4'][:, tissue_index]

        # repeat the omega vector for each material in tissue_type
        omega = np.tile(omega, (1, tissue_number))
        sigma = np.tile(sigma, (omega.shape[0], 1))
        ef = np.tile(ef, (omega.shape[0], 1))
        epsilon = ef + sigma / (1j * omega * self.epsilon0)

        for k in range(0, 4):
            d = np.tile(delta[k, :], (omega.shape[0], 1))
            t = np.tile(tau[k, :], (omega.shape[0], 1))
            a = np.tile(alpha[k, :], (omega.shape[0], 1))
            epsilon = epsilon + d / (1+(1j * omega * t)**(1-a))

        epsilon = epsilon * self.epsilon0

        # if one of the columns is completely 0, this was Lens and needs to
        # computed as mean of LensCortex and LensNucleus
        zero_index = np.nonzero(epsilon == 0)
        if np.count_nonzero(epsilon == 0):
            eps_temp = self.complex_permittivity(self.get_tissue_id_for_name(['LensCortex', 'LensNucleus']), f)
            eps_lens = (eps_temp[:, 0]+eps_temp[:, 1]) / 2
            epsilon[zero_index] = eps_lens

        return epsilon

    def get_tissue_id_for_name(self, tissue_name: Union[str, List[str]]) -> np.ndarray:
        return self.get_id_for_name(tissue_name)

    def get_tissue_name_for_id(self, tissue_index: np.ndarray) -> List[str]:
        return self.get_name_for_id(tissue_index)

    def add_new_tissue(self, tissue_name: str, copy_values_from: Optional[str],
                       new_values: Optional[Dict]=None):
        self.add_new_dielectric(tissue_name, copy_values_from, new_values)

        self.save('ColeColeConstants.TissueProperties')
