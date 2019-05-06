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
An example that shows how to generate a CST voxel model from a VoxelModel.
"""

from os.path import join
from LayerModel_lib import VoxelModel
from config import phantom_path

VoxelModel.working_directory = phantom_path
all_models = VoxelModel.list_all_voxel_models()
for model in all_models:
    print(model)

# Load a virtual human model
vm = VoxelModel('VisibleHuman')

# export the trunk model of this human model
path = join("CST")
vm.export_to_CST("trunk", path)