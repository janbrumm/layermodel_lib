# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import os
import numpy as np

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate

current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'VisHum')

# path to the AVW File of this model
filename = os.path.join(base_path, 'segm_vishum')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportVisibleHuman_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

VisibleHuman = VoxelModel()
VisibleHuman.show_progress_bar = True
# from README.txt
VisibleHuman.set_scale(0.94, 0.91, 5)

VisibleHuman.name = 'VisibleHuman'
VisibleHuman.description = 'Visible Human model from the Helmholtz Zentrum München. ' \
                           'Resolution %.2fmm x %.2fmm x %.2fmm' % (VisibleHuman.scaling.x,
                                                                    VisibleHuman.scaling.y, VisibleHuman.scaling.z)

# Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

VisibleHuman.add_voxel_data(short_name='original',
                            name='Original data from AVW file',
                            model=model_orig,
                            outer_shape=outer_shape,
                            tissue_names=tissue_name_orig)

VisibleHuman.add_voxel_data(short_name='complete',
                            name='The \'original\' model converted to our TissueProperties.',
                            model=VisibleHuman.models['original'].data,
                            outer_shape=outer_shape,
                            tissue_mapping=tissue_mapping)

# calculate the trunk model
# the slice z_end will not be included in the final model
(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(VisibleHuman, model_type='complete', z_start=75, z_end=196)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

VisibleHuman.add_voxel_data(short_name='trunk',
                            name="The trunk of the 'complete' model. Arms have been removed using "
                                 "VoxelModel.remove_arms().",
                            outer_shape=outer_shape_trunk,
                            model=model_trunk,
                            mask=trunk_mask,
                            tissue_mapping=None)

surface = VisibleHuman.create_3d_model(model_type='trunk', patch_size=(30, 30))
VisibleHuman.models['trunk'].surface_3d = surface

VisibleHuman.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    VisibleHuman.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))


VisibleHuman.save_model()
