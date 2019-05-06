# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
"""
This examples shows how to use the LayerModel class to calculate the propagation behaviour
of a plane wave through an arbitrary multi-layered dielectric, where the permittivity and conductivity
follow a simple frequency dependency.
"""
import numpy as np
from typing import Union

from LayerModel_lib import LayerModel, DielectricProperties


class SimpleDielectric(DielectricProperties):
    def complex_permittivity(self, dielectric_index: Union[np.ndarray, float],
                             f: Union[np.ndarray, float]) -> np.ndarray:
        # Here the calculation of epsilon takes place
        epsilon = self.epsilon0 * self.values['eps_real'][dielectric_index] \
                  - 1j * self.values['sigma'][dielectric_index]/(2*np.pi*f)

        return epsilon


# create an object for the dielectric properties
d = SimpleDielectric()
d.add_new_dielectric('Air', new_values={'eps_real': 1, 'sigma': 0})  # index 0 = this should always be Air
d.add_new_dielectric('Solid1', new_values={'eps_real': 3, 'sigma': 5})  # index 1
d.add_new_dielectric('Solid2', new_values={'eps_real': 5, 'sigma': 7})  # index 2

# create a layer model using these dielectric properties
lm = LayerModel.create_from_dict({'Air': None, 'TX': None, 'Solid1': 10, 'Solid2': 20, 'RX': None},
                                 tissue_properties=d)

lm.print_info()
# calculate the transfer function at 1e9 Hz
(transfer_function, frequency) = lm.S21(f_start=1e9, f_end=1e9, n_samples=1)