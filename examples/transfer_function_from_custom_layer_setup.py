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
of a plane wave through arbitrary different tissue layers.
"""
import numpy as np
import matplotlib.pyplot as plt

from LayerModel_lib import LayerModel, TissueProperties

# create a layer model using these dielectric properties
tp = TissueProperties()

# create a layer model with 200 mm of fat layer as source impedance and 10 mm muscle between TX and RX
lm = LayerModel.create_from_dict({'Air': None, 'Fat': 200, 'TX': None, 'Muscle': 10, 'RX': None})
# print an overview over the layer model
lm.print_info()

# calculate the transfer function for S21 at 1e9 Hz
(transfer_function, frequency) = lm.S21(f_start=3.1e9, f_end=4.8e9, n_samples=100)

# plot the magnitude
plt.plot(frequency/1e9, 20*np.log10(np.abs(transfer_function)))
plt.xlabel("Frequency in GHz")
plt.ylabel("Magnitude in dB")
plt.title("Transfer Function of Custom Layer Setup")
plt.show()
