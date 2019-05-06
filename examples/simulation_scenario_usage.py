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
How to use the SimulationScenario class for storing Simulation Results.
"""

import numpy as np
import logging
from datetime import datetime
from os.path import dirname
from sys import stdout

from LayerModel_lib import SimulationScenario, VoxelModel, LayerModel
from config import phantom_path

VoxelModel.working_directory = phantom_path
# turn on logging for more detailed output
logging.basicConfig(level=logging.INFO, stream=stdout)

# the voxel model used in this example
model_name = 'AustinWoman_v2.5_2x2x2'

# Load a virtual human model
vm = VoxelModel(model_name)
# generate 10 random endpoints on the trunk
endpoints = vm.get_random_endpoints('trunk', 10)
# get 10 random startpoints in the gastrointestinal tract
startpoints = vm.get_random_startpoints('trunk', 'GIcontents', 10)
# calculate the layer model from two of these points

# Initialize the SimulationScenario working directory to this example folder
SimulationScenario.working_directory = dirname(__file__)

# create the scenario
scenario = SimulationScenario('create', model_name=model_name, scenario='test')
# add a description
scenario.scenario_description = "This is an example scenario containing some random TX and RX locations."

# add the TX (startpoints) and RX (endpoints) locations
scenario.startpoints = startpoints
scenario.endpoints = endpoints

# compute some results
path_loss = np.zeros(shape=(len(endpoints), len(startpoints)))
for (i, e) in enumerate(endpoints):
    for (k, s) in enumerate(startpoints):
        lm = LayerModel(vm, s, e)
        # calculate the path loss for all of the points
        path_loss[i, k] = lm.path_loss(f_start=3.1e9, f_end=4.8e9, n_samples=1024)

# create a dictionary containing the results
results = scenario.create_results_data_dict(created_on=datetime.today(),
                                            name='PathLoss',
                                            created_by=__file__,
                                            description="Path Loss in the range 3.1-4.8 GHz",
                                            parameters={'f_start': 3.1e9, 'f_end': 4.8e9, 'n_samples': 1024},
                                            readme="The key 'path_loss' contains the path loss results between "
                                                   "each endpoint i and startpoint k in entry path_loss[i, k]")
# add the results to the dictionary
results['path_loss'] = path_loss

# add the results to the scenario
scenario.add_result(results_data=results)

# save the scenario
scenario.save()
