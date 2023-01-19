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
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'Irene')

# path to the AVW File of this model
filename = os.path.join(base_path, 'Irene')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportIrene_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Irene = VoxelModel()
Irene.show_progress_bar = True

# needs to be set manually from README.txt
Irene.set_scale(1.875, 1.875, 5)

Irene.name = 'Irene'
Irene.description = 'Irene model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Irene.scaling.x,
                                                             Irene.scaling.y, Irene.scaling.z)

# For some reason Irene needs to be shifted left and back circularly
# first the shift left
model_orig = np.hstack((model_orig[:, 166::, :], model_orig[:, 0:166, :]))
# then shift back
model_orig = np.vstack((model_orig[16::, :, :], model_orig[0:16, :, :]))

# Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Irene.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Irene.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Irene.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)


# Calculate the trunk model 30:81
start_slice = int(177)
end_slice = int(290)

(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Irene, 'complete', z_start=start_slice, z_end=end_slice)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

Irene.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

surface = Irene.create_3d_model(model_type='trunk', patch_size=(30, 30))
Irene.models['trunk'].surface_3d = surface

Irene.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    Irene.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

# clean up some artifacts that may occur in the generation of the 3d model
Irene.cleanup_3d_model(model_type='trunk', z_max=1300)

# display the 3d model
Irene.show_3d_model('trunk', show_endpoints=True)

Irene.save_model()
