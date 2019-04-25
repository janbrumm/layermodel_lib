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
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate

current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', 'Numerical Human Phantoms unzipped')

# Folder with all the txt-files of the AustinMan model
txt_files_dir = os.path.join(base_path, 'AustinMan-v2_5-2x2x2 txt')

# file with all the tissue ids and the mapping between our TissueProperties and the one in the model
tissue_mapping_file = os.path.join('ImportAustinXXX_TissuesMaterialv2_3.txt')

# Define VoxelModel Object to store all properties of the AustinMan VoxelModel
AustinMan = VoxelModel()
AustinMan.show_progress_bar = True
# Set the name
AustinMan.name = 'AustinMan_v2.5_2x2x2'
AustinMan.description = 'AustinMan v2.5, Resolution: 2mm x 2mm x 2mm'

# get all the data from the txt files
txt_data = VoxelModelImporter(txt_files_dir, tissue_mapping_file, 'TXT', show_progress=True)

# the scale of the model in millimeter
scale = txt_data.data['scale']
AustinMan.set_scale(scale['x'], scale['y'], scale['z'])
# the original voxel model
model_orig = txt_data.data['image']
# the tissue names
tissue_name_orig = txt_data.tissue_names
tissue_mapping = txt_data.tissue_mapping

"""
#The original model
"""
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
AustinMan.add_voxel_data(short_name='complete',
                         name="The 'original' model converted to our TissueProperties.",
                         outer_shape=outer_shape,
                         model=AustinMan.models['original'].data,  # the model of the complete points to the 'original'
                         tissue_mapping=tissue_mapping)
"""
# Calculate the trunk model
"""
# remove the arms, this function makes a copy of the model array. Hence, the tissue indices have been converted to
# our tissue names and it is not necessary to store the tissue_mapping for this model.
(model_trunk, trunk_mask) = txt_data.calculate_trunk_model(AustinMan, model_type='complete', z_start=480, z_end=821)
outer_shape_trunk = txt_data.calculate_outer_shape(model_trunk)

AustinMan.add_voxel_data(short_name='trunk',
                         name="The trunk of the 'complete' model. Arms have been removed using "
                              "VoxelModel.remove_arms().",
                         outer_shape=outer_shape_trunk,
                         model=model_trunk,
                         mask=trunk_mask,
                         tissue_mapping=None)

# create the 3D surface of the voxel model
surface = AustinMan.create_3d_model(model_type='trunk', patch_size=(30, 30))
AustinMan.models['trunk'].surface_3d = surface

AustinMan.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    AustinMan.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

# plot all the patches
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot the surface:
for s in AustinMan.models['trunk'].surface_3d:
    verts = s['verts']
    centroid = s['centroid']
    surf = Poly3DCollection(verts)
    surf.set_facecolor((0, 0, 0, 0.7))
    surf.set_edgecolor('k')
    ax.add_collection3d(surf)
    ax.plot(np.array([centroid[0]]), np.array([centroid[1]]), np.array([centroid[2]]), '.')

ax.set_zlim(800, 1500)
ax.set_aspect('equal', 'box')

plt.show()

# save the model
AustinMan.save_model()
