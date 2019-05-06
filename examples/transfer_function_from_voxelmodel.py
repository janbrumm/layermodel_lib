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
A small example showing how to generate a layer model for specific coordinates inside the chosen VoxelModel.
Additionally, the transfer function for S_21 and E-, and H-field are compared for surface to in-body communication
and vice versa

"""
import numpy as np
import matplotlib.pyplot as plt

from LayerModel_lib import VoxelModel, LayerModel, Coordinate
from config import phantom_path

VoxelModel.working_directory = phantom_path
all_models = VoxelModel.list_all_voxel_models()
for model in all_models:
    print(model)

# Load a virtual human model
vm = VoxelModel('AustinMan_v2.5_2x2x2')

start = Coordinate([231, 270, 1110])  # some point in the colon
end = Coordinate([330, 277, 1110])  # some point outside the body

# calculate the layer model from two of these points
lm = LayerModel(vm, start, end)

# show info about the model
lm.print_info()

# Calculate the transfer function for S21 (square root of transmitted power) from 0 to 10 GHz with 1024 samples,
# the default direction is 'start->end'
(transfer_function, frequency) = lm.S21(f_start=3.1e9, f_end=4.8e9, n_samples=1024)

# plot the magnitude
hf, ha = plt.subplots(nrows=2)
ha[0].plot(frequency / 1e9, 20 * np.log10(np.abs(transfer_function)))
ha[0].set_xlabel("Frequency in GHz")
ha[0].set_ylabel("Magnitude in dB")
ha[0].set_title("Transfer Function between the two coordinates")

ha[1].plot(frequency / 1e9, np.unwrap(np.angle(transfer_function), axis=0))
ha[1].set_xlabel("Frequency in GHz")
ha[1].set_ylabel("Phase in rad")
plt.tight_layout()
plt.show()
