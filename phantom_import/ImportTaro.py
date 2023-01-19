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
import logging
logging.basicConfig(level=logging.INFO)

from os.path import join, dirname

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate

current_directory = dirname(__file__)
base_path = join(current_directory,  '..', '..', '..', 'Numerical Human Phantoms', 'NICT')

# Initialize an empty VoxelModelImporter
importer = VoxelModelImporter('', '', '', show_progress=True)

data_size = 320 * 160 * 804
shape = (320, 160, 804)
voxel_data = np.reshape(np.fromfile(join(base_path, 'Male-v1.raw'), dtype=np.uint8, count=data_size),
                        newshape=shape,
                        order='F')

voxel_size = [2, 2, 2]  # mm size of the voxels

# file with all the tissue ids and the mapping between our TissueProperties and the one in the model
tissue_mapping_file = join('ImportTaro_tissues.txt')

# Define VoxelModel Object to store all properties of the AustinMan VoxelModel
vm = VoxelModel()
vm.show_progress_bar = True
# Set the name
vm.name = 'Taro'
vm.description = 'Taro (NICT), Resolution: %2.2fmm x %2.2fmm x %2.2fmm' % (voxel_size[0],
                                                                             voxel_size[1],
                                                                             voxel_size[2])

# set the scale of the model in millimeter
vm.set_scale(scale_x=voxel_size[1], scale_y=voxel_size[0], scale_z=voxel_size[2])

# The dimensions are originally:
# Data format：1 - byte（256 - ID）
# Direction：
# -X（left），+X（right）
# -Y（back），+Y（front）
# -Z（down），+Z（up）
# They need to be
# 0: Posterior to anterior
# 1: Right to left
# 2: Inferior to superior
# the voxel data needs to be transposed to fit to our coordinate system
model_orig = np.transpose(voxel_data, (1, 0, 2))  # change x and y coordinate
# model_orig = np.flip(model_orig, 1)  # invert the new y-direction left to right
model_orig = np.flip(model_orig, 2)  # invert the z-direction upside down
model_orig = np.flip(model_orig, 0)  # invert the z-direction upside down

# the tissue names
importer.read_tissue_mapping(tissue_mapping_file)
tissue_name_orig = importer.tissue_names
tissue_mapping = importer.tissue_mapping

"""
#The original model
"""
# Calculate the outer_shape
outer_shape = importer.calculate_outer_shape(model_orig, tissue_mapping)

# store the original human body model
vm.add_voxel_data(short_name='original',
                  name='The original model as imported from the Male-v1.raw file of NICT.',
                  outer_shape=outer_shape,
                  model=model_orig,
                  tissue_names=tissue_name_orig)

"""
# the complete model
"""
# add the 'complete' model. This is in fact just a pointer to the 'original' model with the tissue_mapping
# which is automatically applied when the model is called in the following way:
#  Alvar.model['complete'][:, :, slice] gives one horizontal slice of the complete model with our tissue names.
# This is handled by the __getitem__() method of VoxelModelData.
vm.add_voxel_data(short_name='complete',
                  name="The 'original' model converted to our TissueProperties.",
                  outer_shape=outer_shape,
                  model=vm.models['original'].data,  # the model of the complete points to the 'original'
                  tissue_mapping=tissue_mapping)

"""
# Calculate the trunk model
"""
# remove the arms, this function makes a copy of the model array. Hence, the tissue indices have been converted to
# our tissue names and it is not necessary to store the tissue_mapping for this model.
(model_trunk, trunk_mask) = importer.calculate_trunk_model(vm, model_type='complete', z_start=382, z_end=660)
outer_shape_trunk = importer.calculate_outer_shape(model_trunk)

vm.add_voxel_data(short_name='trunk',
                  name="The trunk of the 'complete' model. Arms have been removed using "
                       "VoxelModel.remove_arms().",
                  outer_shape=outer_shape_trunk,
                  model=model_trunk,
                  mask=trunk_mask,
                  tissue_mapping=None)

# create the 3D surface of the voxel model
surface = vm.create_3d_model(model_type='trunk', patch_size=(30, 30))
vm.models['trunk'].surface_3d = surface

vm.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    vm.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

vm = importer.cluster_endpoints(vm)

# clean up some artifacts that may occur in the generation of the 3d model
vm.cleanup_3d_model(model_type='trunk', z_max=None)

# display the 3d model
vm.show_3d_model('trunk', show_endpoints=True)
vm.show_3d_model('trunk', show_clusters=True)

# save the model
vm.save_model()
