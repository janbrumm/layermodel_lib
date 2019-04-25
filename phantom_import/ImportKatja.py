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
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'Katja')

# path to the AVW File of this model
filename = os.path.join(base_path, 'Katja')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportKatja_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Katja = VoxelModel()
Katja.show_progress_bar = True

# needs to be set manually from README.txt
Katja.set_scale(1.775, 1.775, 4.84)

Katja.name = 'Katja'
Katja.description = 'Katja model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Katja.scaling.x,
                                                             Katja.scaling.y, Katja.scaling.z)

# For some reason Katja needs to be shifted left and back circularly
# first the shift left
model_orig = np.hstack((model_orig[:, 210::, :], model_orig[:, 0:210, :]))
# then shift back
model_orig = np.vstack((model_orig[16::, :, :], model_orig[0:16, :, :]))
# Katja has inverse x coordinates in the model. Correct that by flipping along the first axis
model_orig = np.flip(model_orig, axis=0)

# Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Katja.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Katja.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Katja.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)

# Calculate the trunk model
start_slice = int(177)
end_slice = int(290)

(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Katja, 'complete', z_start=start_slice, z_end=end_slice)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

Katja.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

surface = Katja.create_3d_model(model_type='trunk', patch_size=(30, 30))
Katja.models['trunk'].surface_3d = surface

Katja.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    Katja.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

Katja.save_model()
