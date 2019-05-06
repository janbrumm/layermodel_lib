# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
"""
Example that shows how the layer model can be visualized.
"""
from LayerModel_lib import VoxelModel, LayerModel, Coordinate, ModelNames

vm = VoxelModel(model_name=ModelNames.VisibleHuman)

tx = Coordinate([238.29999999999998,  350.35, 690.])
rx = Coordinate([308.79999999999995,  91.0,  630.])

lm = LayerModel(voxel_model=vm, startpoint=tx, endpoint=rx)

lm.plot(title='Layer Model Test')

lm.plot_layer_compare([lm], ['test'])