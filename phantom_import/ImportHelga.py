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
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'Helga')

# path to the AVW File of this model
filename = os.path.join(base_path, 'segm_helga')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportHelga_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Helga = VoxelModel()
Helga.show_progress_bar = True

# needs to be set manually from README.txt
Helga.set_scale(0.98, 0.98, 10)

Helga.name = 'Helga'
Helga.description = 'Helga model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Helga.scaling.x,
                                                             Helga.scaling.y, Helga.scaling.z)

# Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Helga.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Helga.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Helga.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)

# Calculate the trunk model
(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Helga, 'complete', z_start=30, z_end=81)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

Helga.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

surface = Helga.create_3d_model(model_type='trunk', patch_size=(30, 30))
Helga.models['trunk'].surface_3d = surface

Helga.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    Helga.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

Helga.save_model()