# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Hannah Strohm, Jan-Christoph Brumm
#
# Licensed under MIT license.
#
"""
This script sets the endpoints (RX locations on the abdomen) and clusters them into 16 clusters.
These are numbered as follows:
         A1 | A2 | B1 | B2 |
         A3 | A4 | B3 | B4 |
         C1 | C2 | D1 | D2 |
         C3 | C4 | D3 | D4 |
(this is the view of the abdominal surface from the front)
Where the first number (A, B, C, D) gives a coarse position and the second number (1, 2, 3, 4) gives the finer position
inside the coarse cluster.

The endpoints on the abdomen as well as the mapping to the clusters is added to the VoxelModel.models['trunk'] model as
    - endpoints_abdomen : a list of receiver locations
    - endpoints_abdomen_clustering : the mapping
"""

import numpy as np

from os.path import join
from LayerModel_lib import VoxelModel
VoxelModel.working_directory = join('..', 'phantoms')


def cluster_endpoints(endpoints: np.array):
    """
    Function to split up the endpoint-array in four clusters
    :param endpoints: sorted array of endpoints which should be clustered
    :return: the endpoints sorted in four clusters
            cluster mapper: array with the same length as the original endpoint array to indicate which point is in
            which cluster: 0 --> first cluster, 1 --> second cluster, ...
    """

    max_z = np.max(endpoints[:, 2])
    min_z = np.min(endpoints[:, 2])
    half_z = max_z - ((max_z - min_z) / 2)

    max_y = np.max(endpoints[:, 1])
    min_y = np.min(endpoints[:, 1])
    half_y = max_y - ((max_y - min_y) / 2)

    clusterA = list()
    clusterB = list()
    clusterC = list()
    clusterD = list()
    cluster_mapper = np.zeros((len(endpoints), 1))

    # fist approach to sort the endpoints in four clusters
    # | A | B |
    # | C | D |
    for e in endpoints:
        # cluster A
        if e[1] <= half_y and e[2] >= half_z:
            clusterA.append(e)
        # cluster B
        elif e[1] > half_y and e[2] >= half_z:
            clusterB.append(e)
        # cluster C
        elif e[1] <= half_y and e[2] < half_z:
            clusterC.append(e)
        # cluster D
        elif e[1] > half_y and e[2] < half_z:
            clusterD.append(e)

    clusterA_array = np.array(clusterA)
    clusterB_array = np.array(clusterB)
    clusterC_array = np.array(clusterC)
    clusterD_array = np.array(clusterD)

    # compensate between A and C in case the endpoints are badly distributed
    diff_A_C = len(clusterA) - len(clusterC)
    if diff_A_C > 0:  # cluster A has more endpoints than cluster C
        last_z = clusterA_array[len(clusterA) - 1][2] # z-value of last endpoint in A
        lr = 1
        counter = len(clusterA)-2
        # count the endpoints with the same z-value as the last endpoint (10cm difference are allowed) --> this points
        # lie in a row
        while(clusterA_array[counter][2] < last_z + 10):
            lr += 1
            counter -= 1
        if np.abs(diff_A_C - 2*lr) < diff_A_C:  # compensation is advantageous
            comp = clusterA_array[len(clusterA_array) - lr: len(clusterA_array)]
            clusterC_array = np.concatenate((comp, clusterC_array)) # last row from A is concatenated with C
            clusterA_array = clusterA_array[0:len(clusterA_array)-lr] # A is shortened
    elif diff_A_C < 0:  # cluster C has more endpoints than cluster A
        first_z = clusterC_array[0][2] # z-value of first endpoint in C
        fr = 1
        counter = 1
        while(clusterC_array[counter][2] > first_z -10):
            fr += 1
            counter += 1
        if np.abs(diff_A_C + 2*fr) < np.abs(diff_A_C):  # compensation is advantageous
            comp = clusterC_array[0: fr]
            clusterA_array = np.concatenate((clusterA_array, comp))  # the row from C is concatenated with A
            clusterC_array = clusterC_array[fr:]

    # compensate between B and D
    diff_B_D = len(clusterB) - len(clusterD)
    if diff_B_D > 0:  # cluster B has more endpoints than cluster D
        last_z = clusterB_array[len(clusterB) - 1][2] # z-value of last endpoint in B
        lr = 1
        counter = len(clusterB)-2
        # count the endpoints with the same z-value as the last endpoint (10cm difference are allowed) --> this points
        # lie in a row
        while(clusterB_array[counter][2] < last_z + 10):
            lr += 1
            counter -= 1
        if np.abs(diff_B_D - lr) < diff_B_D:  # compensation is advantageous
            comp = clusterB_array[len(clusterB_array) - lr: len(clusterB_array)]
            clusterD_array = np.concatenate((comp, clusterD_array))
            clusterB_array = clusterB_array[0:len(clusterB_array)-lr]
    elif diff_B_D < 0:  # cluster D has more endpoints than cluster B
        first_z = clusterD_array[0][2] # z-value of first endpoint in D
        fr = 1
        counter = 1
        while(clusterD_array[counter][2] > first_z -10):
            fr += 1
            counter += 1
        if np.abs(diff_B_D + fr) < np.abs(diff_B_D):  # compensation is advantageous
            comp = clusterD_array[0: fr]
            clusterB_array = np.concatenate((clusterB_array, comp))  # the row from D is concatenated with B
            clusterD_array = clusterD_array[fr:]

    # set up the cluster mapper
    for (e,i) in zip(endpoints, range(len(endpoints))):
        e = e.tolist()
        if e in clusterA_array.tolist():
            cluster_mapper[i,:] = 0
        elif e in clusterB_array.tolist():
            cluster_mapper[i,:] = 1
        elif e in clusterC_array.tolist():
            cluster_mapper[i,:] = 2
        elif e in clusterD_array.tolist():
            cluster_mapper[i,:] = 3

    return clusterA_array, clusterB_array, clusterC_array, clusterD_array, cluster_mapper


if __name__ == '__main__':
    # Get a list of all models and run the script for them all
    all_models = ['AustinMan_v2.5_2x2x2',
                  'AustinMan_v2.6_1x1x1',
                  'AustinWoman_v2.5_1x1x1',
                  'AustinWoman_v2.5_2x2x2',
                  'Donna',
                  'Golem',
                  'Helga',
                  'Irene',
                  'Katja',
                  'VisibleHuman']
    for model_name in all_models:
        vm = VoxelModel(model_name)

        # defined ranges for the x- and z-Parameters to get endpoints only at the front (belly)
        x_ranges = {'AustinWoman_v2.5_2x2x2': (230,350),'AustinWoman_v2.5_1x1x1': (230,350),
                    'AustinMan_v2.5_2x2x2': (250, 350), 'AustinMan_v2.6_1x1x1': (250, 350),
                    'Irene': (150,200), 'Donna': (320,400), 'Golem': (220,350), 'Helga': (300,400),
                    'VisibleHuman': (300,450), 'Katja': (175,300), 'Frank': (200,380)}

        z_ranges = {'AustinWoman_v2.5_2x2x2': [900,1250],'AustinWoman_v2.5_1x1x1': [900,1220],
                    'AustinMan_v2.5_2x2x2': [1000,1350], 'AustinMan_v2.6_1x1x1':[1000,1350],
                    'Irene': (950,1280), 'Donna': (1000,1300), 'Golem': (900,1270), 'Helga': (330,685),
                    'VisibleHuman': (380,750), 'Katja': (910,1220), 'Frank': (100,400)}

        # navel for the models set by hand set by hand
        navel = {'AustinWoman_v2.5_2x2x2': np.array([ 305.9999694,  359.999964 , 1074.]),
                 'AustinWoman_v2.5_1x1x1': np.array([ 309.999969 ,  366.9999633, 1054.]),
                 'AustinMan_v2.5_2x2x2': np.array( [325.9999674,  363.9999636, 1124]),
                 'AustinMan_v2.6_1x1x1':  np.array([319.999968 ,  352.9999647, 1129]),
                 'Irene': np.array([ 185.625,  243.75 , 1105.]),
                 'Donna': np.array( [ 378.75 ,  215.625, 1150.]),
                 'Golem': np.array([ 280.8 ,  257.92, 1080]),
                 'Helga': np.array([363.58, 238.14, 490.]),
                 'VisibleHuman': np.array([389.16, 228.41, 535.]),
                 'Katja': np.array( [ 255.6  ,268.025, 1045.44 ]),
                 'Frank': np.array([0,0,0])}

        # from all endpoints on the trunk pick the ones at the front
        # (defined range of x,y- and z-Parameters, set by hand for each model)
        endpoints = vm.models['trunk'].endpoints
        num_endpoints = np.shape(endpoints)[0]
        e = list()

        for i in range(num_endpoints):
            if x_ranges[model_name][0] <= endpoints[i].x <= x_ranges[model_name][1] and \
                    z_ranges[model_name][0] <= endpoints[i].z <= z_ranges[model_name][1] :
                e.append(endpoints[i])

        # Sort the points: z in descending and y in ascending order such that the points start at the upper right corner
        endpoints_sorted = sorted(e, key = lambda e: (-e.get_z(), e.get_y()))
        endpoints_sorted_array = np.array(endpoints_sorted)

        # Cluster Mapping
        # split the endpoints in four regions
        clusterA, clusterB, clusterC, clusterD, cluster_map_1st = cluster_endpoints(endpoints_sorted_array)

        # subclustering: split the above clusters again into four regions
        clusterA1, clusterA2, clusterA3, clusterA4, cluster_mapA = cluster_endpoints(clusterA)
        clusterB1, clusterB2, clusterB3, clusterB4, cluster_mapB = cluster_endpoints(clusterB)
        clusterC1, clusterC2, clusterC3, clusterC4, cluster_mapC = cluster_endpoints(clusterC)
        clusterD1, clusterD2, clusterD3, clusterD4, cluster_mapD = cluster_endpoints(clusterD)

        # For each endpoint find the subcluster and set up a cluster map for the 16 clusters
        # cluster_map: defines overall mapping
        # array of shape [len(endpoints),2] which indicates for each endpoint the cluster to which it belongs
        # A1 | A2 | B1 | B2 |
        # A3 | A4 | B3 | B4 |
        # C1 | C2 | D1 | D2 |
        # C3 | C4 | D3 | D4 |
        # first number indicates cluster A,B,C,D; second number the subcluster 1,2,3,4

        cluster_map = np.concatenate((cluster_map_1st, np.zeros((len(cluster_map_1st),1))), axis = 1)
        for (e,i) in zip(endpoints_sorted_array, range(len(endpoints_sorted))):
            e_list = e.tolist()
            if e_list in clusterA.tolist():
                clusterA_list = clusterA.tolist()
                index = clusterA_list.index(e_list)
                cluster_map[i,1] = cluster_mapA[index,0]
            if e_list in clusterB.tolist():
                clusterB_list = clusterB.tolist()
                index = clusterB_list.index(e_list)
                cluster_map[i,1] = cluster_mapB[index,0]
            if e_list in clusterC.tolist():
                clusterC_list = clusterC.tolist()
                index = clusterC_list.index(e_list)
                cluster_map[i,1] = cluster_mapC[index,0]
            if e_list in clusterD.tolist():
                clusterD_list = clusterD.tolist()
                index = clusterD_list.index(e_list)
                cluster_map[i,1] = cluster_mapD[index,0]

        # add the endpoints and the cluster mapping as variables to the truncated model
        vm.models['trunk'].endpoints_abdomen = endpoints_sorted
        vm.models['trunk'].endpoints_abdomen_clustering = cluster_map

        # save the model
        vm.save_model()

