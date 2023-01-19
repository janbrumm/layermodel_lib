# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import os
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate

current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', 'Numerical Human Phantoms unzipped')

# Folder with all the txt-files of the AustinMan model
txt_files_dir = os.path.join(base_path, 'AustinMan-v2_6-1x1x1')

# file with all the tissue ids and the mapping between our TissueProperties and the one in the model
tissue_mapping_file = os.path.join('ImportAustinXXX_TissuesMaterialv2_3.txt')

# Define VoxelModel Object to store all properties of the AustinMan VoxelModel
print("Creating VoxelModel...")
AustinMan = VoxelModel()
AustinMan.show_progress_bar = True
# Set the name
AustinMan.name = 'AustinMan_v2.6_1x1x1'
AustinMan.description = 'AustinMan v2.6, Resolution: 1mm x 1mm x 1mm'

# get all the data from the txt files
print("Import the data...")
txt_data = VoxelModelImporter(txt_files_dir, tissue_mapping_file, 'TXT', show_progress=True)
# the scale of the model in millimeter

scale = txt_data.data['scale']
AustinMan.set_scale(scale['x'], scale['y'], scale['z'])
model_orig = txt_data.data['image']
tissue_name_orig = txt_data.tissue_names
tissue_mapping = txt_data.tissue_mapping

"""
#The original model
"""
print("Save the original data...")
# Calculate the outer_shape
outer_shape = txt_data.calculate_outer_shape(model_orig, tissue_mapping)

# store the original human body model
AustinMan.add_voxel_data(short_name='original',
                         name='The original model as imported from the .txt files.',
                         outer_shape=outer_shape,
                         model=model_orig,
                         tissue_names=tissue_name_orig)
"""
# the complete model
"""
# add the 'complete' model. This is in fact just a pointer to the 'original' model with the tissue_mapping
# which is automatically applied when the model is called in the following way:
#  AustinMan.model['complete'][:, :, slice] gives one horizontal slice of the complete model with our tissue names.
# This is handled by the __getitem__() method of VoxelModelData.
print("Save the 'complete' model using the tissue_mapping to TissueProperties...")
AustinMan.add_voxel_data(short_name='complete',
                         name="The 'original' model converted to our TissueProperties.",
                         outer_shape=outer_shape,
                         model=AustinMan.models['original'].data,  # the model of the complete points to the 'original'
                         tissue_mapping=tissue_mapping)

"""
# Calculate the trunk model
"""
# The start and end slice of the trunk model:
start_slice = int(965)
end_slice = int(1591)

# remove the arms, this function makes a copy of the model array. Hence, the tissue indices have been converted to
# "our" tissue names and it is not necessary to store the tissue_mapping for this model.
(model_trunk, trunk_mask) = txt_data.calculate_trunk_model(AustinMan, model_type='complete', z_start=start_slice,
                                                           z_end=end_slice)
outer_shape_trunk = txt_data.calculate_outer_shape(model_trunk)

AustinMan.add_voxel_data(short_name='trunk',
                         name="The trunk of the 'complete' model. Arms have been removed using "
                              "VoxelModel.remove_arms().",
                         outer_shape=outer_shape_trunk,
                         model=model_trunk,
                         mask=trunk_mask,
                         tissue_mapping=None)

print("Create the 3D surface of the skin surface..")
surface = AustinMan.create_3d_model(model_type='trunk', patch_size=(30, 30))
AustinMan.models['trunk'].surface_3d = surface

AustinMan.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    AustinMan.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

# clean up some artifacts that may occur in the generation of the 3d model
AustinMan.cleanup_3d_model(model_type='trunk', z_max=1400)

# display the 3d model
AustinMan.show_3d_model('trunk', show_endpoints=True)

# save the model
AustinMan.save_model()
print("Complete model saved successfully..")
