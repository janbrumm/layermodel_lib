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
from numpy.polynomial import Polynomial
from functools import partial
import h5py
from typing import Callable, Union
from os.path import dirname, join


class PolynomialChannelModel:
    """
    The simplified channel model for the channel transfer function in the form of
    H(f) = p_7(f) * exp(j* p_3(f))
    """
    def __init__(self, model: str):
        self.model = model
        self.current_directory = dirname(__file__)
        # open the file
        self.mag_poly = None
        self.phase_poly = None

    # def __del__(self):
    #     # close the file again
    #     self.h5_file.close()

    @property
    def num_tx(self):
        h5_file = h5py.File(join(self.current_directory, 'channel_model.h5'), 'r')
        base_path = f'{self.model}/air/'  # TX RX dimensions should be the same for air and effective tissue
        return h5_file[base_path + 'mag'].shape[0]

    @property
    def num_rx(self):
        h5_file = h5py.File(join(self.current_directory, 'channel_model.h5'), 'r')
        base_path = f'{self.model}/air/'  # TX RX dimensions should be the same for air and effective tissue
        return h5_file[base_path + 'mag'].shape[1]

    def _poly(self, f: Union[np.ndarray, float], negate_f: bool = False) -> np.ndarray:
        """
        The polynomial that is the best fit to the transfer functions.
        """
        if negate_f:
            return np.conjugate(self.mag_poly(-f / 1e9) * np.exp(1j * self.phase_poly(-f / 1e9)))
        else:
            return self.mag_poly(f / 1e9) * np.exp(1j * self.phase_poly(f / 1e9))

    def _tf(self, f: np.ndarray, clipping: bool) -> np.ndarray:
        f = f.astype(complex)
        if clipping:
            return np.piecewise(f, condlist=[np.abs(f) < 3.1e9, np.abs(f) > 4.8e9,
                                             np.logical_and(f >= 3.1e9, f <= 4.8e9),
                                             np.logical_and(f <= -3.1e9, f >= -4.8e9)],
                                funclist=[self._poly(3.1e9), self._poly(4.8e9),
                                          partial(self._poly, negate_f=False),
                                          partial(self._poly, negate_f=True)])
        else:
            return np.piecewise(f, condlist=[f < 0, f >= 0],
                                funclist=[partial(self._poly, negate_f=True), partial(self._poly, negate_f=False)])

    def transfer_function(self, radiation_loss: str,
                          tx_idx: int, rx_idx: int,
                          clipping: bool = True) -> Callable[[np.ndarray], np.ndarray]:
        """
        Return a callable  transfer function of the polynomial channel model.
        :param radiation_loss: The type of radiation loss, either 'air' or 'effective_tissue'
        :param tx_idx: Index of the transmitter location.
        :param rx_idx: Index of the receiver location.
        :param clipping: If set to True (default) the value of the transfer function will be continued with the
                         values from 3.1e9 or 4.8e9 for smaller and larger values, respectively.

        :return:
        """
        h5_file = h5py.File(join(self.current_directory, 'channel_model.h5'), 'r')

        base_path = f'{self.model}/{radiation_loss}/'
        mag_coefficients = h5_file[base_path + 'mag'][tx_idx, rx_idx, :]
        self.mag_poly = Polynomial(coef=mag_coefficients)

        phase_coefficients = h5_file[base_path + 'phase'][tx_idx, rx_idx, :]
        self.phase_poly = Polynomial(coef=phase_coefficients)

        return partial(self._tf, clipping=clipping)
