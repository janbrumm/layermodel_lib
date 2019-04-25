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
from LayerModel_lib.coordinate import *
from LayerModel_lib.dielectric import *
from LayerModel_lib.layer_model import *
from LayerModel_lib.simulation_scenario import *
from LayerModel_lib.tissue_properties import *
from LayerModel_lib.voxelmodel import *
from LayerModel_lib.voxelmodel_importer import *
from LayerModel_lib.voxelmodel_simulator import *
from LayerModel_lib.__version__ import __version__

__author__ = 'Jan-Christoph Brumm <jan.brumm@tuhh.de>'
__description__ = 'A Python tool to compute the transmission behaviour of plane electromagnetic waves ' \
                  'through human tissue.'
__email__ = 'jan.brumm@tuhh.de'
__license__ = 'MIT'
__modulename__ = 'LayerModel_lib'
__website__ = 'https://collaborating.tuhh.de/int/in-body/LayerModel_lib'


class ModelNames:
    # A list of all available VoxelModels (all the ones for which an import script is available)
    AustinMan_v25_2mm = 'AustinMan_v2.5_2x2x2'
    AustinMan_v26_1mm = 'AustinMan_v2.6_1x1x1'
    AustinWoman_v25_1mm = 'AustinWoman_v2.5_1x1x1'
    AustinWoman_v25_2mm = 'AustinWoman_v2.5_2x2x2'
    Donna = 'Donna'
    Frank = 'Frank'
    Golem = 'Golem'
    Helga = 'Helga'
    Irene = 'Irene'
    Katja = 'Katja'
    VisibleHuman = 'VisibleHuman'

    all = [AustinMan_v25_2mm,
           AustinMan_v26_1mm,
           AustinWoman_v25_1mm,
           AustinWoman_v25_2mm,
           Donna,
           Golem,
           Helga,
           Irene,
           VisibleHuman]
