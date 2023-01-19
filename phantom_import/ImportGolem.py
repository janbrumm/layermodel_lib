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
from scipy import ndimage
import logging
logging.basicConfig(level=logging.INFO)

from LayerModel_lib import VoxelModel, VoxelModelImporter, Coordinate, TissueProperties

current_directory = os.path.dirname(__file__)
base_path = os.path.join(current_directory, '..', '..', '..', '..', '..', 'Numerical Human Phantoms', 'Golem')

# path to the AVW File of this model
filename = os.path.join(base_path, 'segm_golem')
# path to the tissue_mapping file
tissue_file = os.path.join('ImportGolem_tissues.txt')

AVW_Data = VoxelModelImporter(filename, tissue_file, 'AVW')
model_orig = AVW_Data.data['image']
tissue_name_orig = AVW_Data.tissue_names
tissue_mapping = AVW_Data.tissue_mapping

Golem = VoxelModel()
Golem.show_progress_bar = True

# needs to be set manually from README.txt
Golem.set_scale(2.08, 2.08, 8)

Golem.name = 'Golem'
Golem.description = 'Golem model from the Helmholtz Zentrum München. ' \
                    'Resolution %.2fmm x %.2fmm x %.2fmm' % (Golem.scaling.x,
                                                             Golem.scaling.y, Golem.scaling.z)

# Golem is upside down and left right in wrong direction
model_orig = np.flip(model_orig, axis=0)
model_orig = np.flip(model_orig, axis=2)
# The space around the model is too much
model_orig = model_orig[0:166, :, :]

#  Calculate the outer_shape of the original and the complete model
outer_shape = AVW_Data.calculate_outer_shape(model_orig, tissue_mapping)

Golem.add_voxel_data(short_name='original',
                     name='Original data from AVW file',
                     model=model_orig,
                     outer_shape=outer_shape,
                     tissue_names=tissue_name_orig)

Golem.add_voxel_data(short_name='complete',
                     name='The \'original\' model converted to our TissueProperties.',
                     model=Golem.models['original'].data,
                     outer_shape=outer_shape,
                     tissue_mapping=tissue_mapping)

# Calculate the trunk model
start_slice = int(110)
end_slice = int(175)

(model_trunk, trunk_mask) = AVW_Data.calculate_trunk_model(Golem, 'complete', z_start=start_slice, z_end=end_slice)
outer_shape_trunk = AVW_Data.calculate_outer_shape(model_trunk)

# label the inner parts of the small intestine as GIcontents. For that erode the small intestine in
# 3D and replace the resulting voxels with GIcontents.
tp = TissueProperties()

trunk = model_trunk

# Golem has no labeled SmallIntestineContents. Therefore, we need to label that by ourself:
# select only the small intenstine
trunk_si = trunk == tp.get_tissue_id_for_name(['SmallIntestine'])
struct = ndimage.generate_binary_structure(3, 1)
# Erode the small intestine wall:
trunk_si_eroded = ndimage.binary_erosion(trunk_si, structure=struct, iterations=1)
trunk_si_out = np.copy(trunk)
# replace remaining inner small intestine with GIcontents
trunk_si_out[trunk_si_eroded] = tp.get_tissue_id_for_name(['GIcontents'])

model_trunk = trunk_si_out

Golem.add_voxel_data(short_name='trunk',
                     name="The trunk of the 'complete' model. Arms have been removed using "
                          "VoxelModel.remove_arms().",
                     outer_shape=outer_shape_trunk,
                     model=model_trunk,
                     mask=trunk_mask,
                     tissue_mapping=None)

surface = Golem.create_3d_model(model_type='trunk', patch_size=(30, 30))
Golem.models['trunk'].surface_3d = surface

Golem.models['trunk'].endpoints = []
for (i, s) in enumerate(surface):
    Golem.models['trunk'].endpoints.append(Coordinate(np.array(s['centroid'])))

# clean up some artifacts that may occur in the generation of the 3d model
Golem.cleanup_3d_model(model_type='trunk', z_max=1300)

# display the 3d model
Golem.show_3d_model('trunk', show_endpoints=True)

Golem.save_model()
