# This file is part of LayerModel_lib
#     A tool to compute the transmission behaviour of plane waves 
#     through human tissue. 
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
A small example showing how to generate a layer model for specific coordinates inside 
the chosen VoxelModel. Additionally, the impulse response for S_21, as well as E-, and H-field 
are compared for surface to in-body communication and vice versa
"""
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

# Calculate the impulse response for fs=20 GHz and 1024 samples
(impulse_S21_out, t_S21_out) = lm.impulse_response(f_sample=20e9, n_samples=1024, direction='start->end')

# plot the magnitude
hf, ha = plt.subplots(nrows=1)
ha.plot(t_S21_out / 1e-9, impulse_S21_out)

ha.set_xlabel("Time in ns")
ha.set_ylabel("Amplitude of h(t)")
ha.legend(loc='upper right', ncol=1)
plt.tight_layout()
plt.show()
