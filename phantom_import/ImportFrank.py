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
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'Frank')

# path to the AVW File of this model
filename = os.path.join(base_path, 'segm_frank')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportFrank_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Frank = VoxelModel()
Frank.show_progress_bar = True

# needs to be set manually from README.txt
Frank.set_scale(0.742188, 0.742188, 5)

Frank.name = 'Frank'
Frank.description = 'Frank model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Frank.scaling.x,
                                                             Frank.scaling.y, Frank.scaling.z)

# For some reason Frank needs to be shifted back circularly
model_orig = np.vstack((model_orig[25::, :, :], model_orig[0:25, :, :]))

#  Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Frank.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Frank.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Frank.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)

# Calculate the trunk model
start_slice = int(0)
end_slice = int(110)

(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Frank, 'complete', z_start=start_slice, z_end=end_slice)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

Frank.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

surface = Frank.create_3d_model(model_type='trunk', patch_size=(30, 30))
Frank.models['trunk'].surface_3d = surface

Frank.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    Frank.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

Frank.save_model()
