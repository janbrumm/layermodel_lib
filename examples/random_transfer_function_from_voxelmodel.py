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
A small example showing how to generate a layer model for random coordinates inside the chosen VoxelModel
"""
import numpy as np
import matplotlib.pyplot as plt

from LayerModel_lib import VoxelModel, LayerModel
from config import phantom_path

VoxelModel.working_directory = phantom_path
all_models = VoxelModel.list_all_voxel_models()
for model in all_models:
    print(model)

# Load a virtual human model
vm = VoxelModel('AustinWoman_v2.5_2x2x2')

# generate 10 random endpoints on the trunk
e = vm.get_random_endpoints('trunk', 10)

# get 10 random startpoints in the gastrointestinal tract
s = vm.get_random_startpoints('trunk', 'GIcontents', 10)

# calculate the layer model from two of these points
lm = LayerModel(vm, s[1], e[1])

# compute the transfer function
transfer_function, f = lm.transfer_function()

# plot the magnitude
plt.plot(f/1e9, 20*np.log10(np.abs(transfer_function)))
plt.xlabel("Frequency in GHz")
plt.ylabel("Magnitude in dB")
plt.title("Transfer Function of two Random Locations")
plt.show()
