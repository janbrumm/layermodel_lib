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
A small example showing how to visualize one transverse slice of the voxel models.
"""
import matplotlib.pyplot as plt

from LayerModel_lib import VoxelModel
from config import phantom_path

VoxelModel.working_directory = phantom_path

vm = VoxelModel('VisibleHuman')

# determine the minimum and maximum values used in the voxel model for proper scaling of the colormap
v_min = vm.models['trunk'].min_tissue_id
v_max = vm.models['trunk'].max_tissue_id

# calculate the extent of the image and scale it according to the scaling of the model
right = max(vm.models['trunk'].mask['y']) * vm.scaling.y
left = vm.models['trunk'].mask['y'].start * vm.scaling.y
bottom = max(vm.models['trunk'].mask['x']) * vm.scaling.x
top = vm.models['trunk'].mask['x'].start * vm.scaling.x
extent = (left, right, bottom, top)

axes_image = plt.imshow(vm.models['trunk'][:, :, 50],
                        cmap=vm.colormap, vmin=v_min, vmax=v_max,
                        extent=extent, origin='upper')

plt.show()