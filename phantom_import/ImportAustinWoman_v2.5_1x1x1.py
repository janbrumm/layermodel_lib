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
import os

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate


current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', 'Numerical Human Phantoms unzipped')

# Folder with all the txt-files of the AustinWoman model
txt_files_dir = os.path.join(base_path, 'AustinWoman-v2_5-1x1x1')

# file with all the tissue ids and the mapping between our TissueProperties and the one in the model
tissue_mapping_file = os.path.join('ImportAustinXXX_TissuesMaterialv2_3.txt')

# Define VoxelModel Object to store all properties of the AustinMan VoxelModel
AustinWoman = VoxelModel()
AustinWoman.show_progress_bar = True
# Set the name
AustinWoman.name = 'AustinWoman_v2.5_1x1x1'
AustinWoman.description = 'AustinWoman v2.5, Resolution: 1mm x 1mm x 1mm'

# get all the data from the txt files
txt_data = VoxelModelImporter(txt_files_dir, tissue_mapping_file, 'TXT', show_progress=True)

scale = txt_data.data['scale']
AustinWoman.set_scale(scale['x'], scale['y'], scale['z'])
model_orig = txt_data.data['image']
tissue_name_orig = txt_data.tissue_names
tissue_mapping = txt_data.tissue_mapping
"""
the original model
"""
outer_shape = txt_data.calculate_outer_shape(model_orig, tissue_mapping)

# store the original human body model
AustinWoman.add_voxel_data(short_name='original',
                           name='The original model as imported from the .txt files.',
                           outer_shape=outer_shape,
                           model=model_orig,
                           tissue_names=tissue_name_orig,
                           tissue_mapping=None)

"""
the complete model
"""
# add the 'complete' model. This is in fact just a pointer to the 'original' model with the tissue_mapping
# which is automatically applied when the model is called in the following way:
#  AustinWoman.model['complete'][:, :, slice] gives one horizontal slice of the complete model with our tissue names.
# This is handled by the __getitem__() method of VoxelModelData.
AustinWoman.add_voxel_data(short_name='complete',
                           name="The 'original' model converted to our TissueProperties.",
                           outer_shape=outer_shape,
                           model=AustinWoman.models['original'].data,
                           tissue_mapping=tissue_mapping)


"""
Calculate the trunk model
"""
(model_trunk, trunk_mask) = txt_data.calculate_trunk_model(AustinWoman, model_type='complete', z_start=830, z_end=1451)
outer_shape_trunk = txt_data.calculate_outer_shape(model_trunk)

AustinWoman.add_voxel_data(short_name='trunk',
                           name="The trunk of the 'complete' model. Arms have been removed using "
                                "VoxelModel.remove_arms().",
                           outer_shape=outer_shape_trunk,
                           model=model_trunk,
                           mask=trunk_mask,
                           tissue_mapping=None)

surface = AustinWoman.create_3d_model(model_type='trunk', patch_size=(30, 30))
AustinWoman.models['trunk'].surface_3d = surface
AustinWoman.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    AustinWoman.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))


# save the model
AustinWoman.save_model()

