# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#

# These lines allow the direct import of the classes defined in the respective files
from LayerModel_lib.channel_model import PolynomialChannelModel
from LayerModel_lib.coordinate import *
from LayerModel_lib.dielectric import *
from LayerModel_lib.layer_model import *
from LayerModel_lib.simulation_scenario import *
from LayerModel_lib.tissue_properties import *
from LayerModel_lib.voxelmodel import *
from LayerModel_lib.voxelmodel_importer import *
from LayerModel_lib.voxelmodel_simulator import *
from LayerModel_lib.general import ModelNames
from LayerModel_lib.__version__ import __version__

phantoms = ModelNames()

__author__ = 'Jan-Christoph Brumm <jan.brumm@tuhh.de>'
__description__ = 'A Python tool to compute the transmission behaviour of plane electromagnetic waves ' \
                  'through human tissue.'
__email__ = 'jan.brumm@tuhh.de'
__license__ = 'MIT'
__modulename__ = 'LayerModel_lib'
__website__ = 'https://collaborating.tuhh.de/int/in-body/LayerModel_lib'


