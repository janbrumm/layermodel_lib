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
import logging
logging.basicConfig(level=logging.INFO)

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.voxelmodel_importer import VoxelModelImporter
from LayerModel_lib.coordinate import Coordinate

current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', '..', '..', 'Numerical Human Phantoms', 'Donna')

# path to the AVW File of this model
filename = os.path.join(base_path, 'segm_donna')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportDonna_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Donna = VoxelModel()
Donna.show_progress_bar = True

# needs to be set manually from README.txt
Donna.set_scale(1.875, 1.875, 10)

Donna.name = 'Donna'
Donna.description = 'Donna model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Donna.scaling.x,
                                                             Donna.scaling.y, Donna.scaling.z)

#  Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Donna.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Donna.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Donna.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)

# Calculate the trunk model
start_slice = int(96)
end_slice = int(141)

(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Donna, 'complete', z_start=start_slice, z_end=end_slice)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

Donna.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

# Calculate the 3D surface of Donna
surface = Donna.create_3d_model(model_type='trunk', patch_size=(30, 30))
Donna.models['trunk'].surface_3d = surface

Donna.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
   Donna.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))#

# clean up some artifacts that may occur in the generation of the 3d model
Donna.cleanup_3d_model(model_type='trunk', z_max=None)

# display the 3d model
Donna.show_3d_model('trunk', show_endpoints=True)

Donna.save_model()
