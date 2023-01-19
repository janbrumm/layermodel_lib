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
import logging
from typing import Dict, Tuple
from os import listdir
from os.path import isfile, join

from typing import Optional

from LayerModel_lib.tissue_properties import TissueProperties
from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.general import ProgressBarConfig as pb


class VoxelModelImporter:
    """
    A class to import the AustinMan/AustinWoman models in txt-format or the models provided by the
    Helmholtz Zentrum München.
    """

    def __init__(self, data_filename: str, tissue_mapping_file: str, file_type: str,
                 show_progress: Optional[bool]=False):
        """
        Create a new importing instance
        :param data_filename:
        :param tissue_mapping_file: A file which describes the tissue mapping to our tissue IDs
        :param file_type:
        :param show_progress:
        """
        self.data = {}
        self.tissue_names = []
        self.tissue_mapping = np.zeros(shape=(0,))

        # can be activated to show a progress bar in the console
        self.show_progress_bar = show_progress

        if file_type == 'AVW':
            self.read_tissue_mapping(tissue_mapping_file)
            self.load_avw_file(data_filename)
        elif file_type == 'TXT':
            self.read_tissue_mapping(tissue_mapping_file)
            self.load_from_txt_files(data_filename)

    @staticmethod
    def calculate_outer_shape(model: np.ndarray, tissue_mapping: np.ndarray=None) -> Dict:
        """
        Calculate the outer_shape of the model, using the optional tissue_mapping to find all skin tissues.

        :param model: np.ndarray containing the voxel data
        :param tissue_mapping:  np.ndarray defining the mapping of tissue IDs in the model
                                to the tissue IDs in TissueProperties
        :return: Dictionary containing the 'front' and 'right' view of the contour of the model.
        """
        # get the ids of all the tissues which contain skin in their name
        skin_ids_TissueProperties = [i for (i, s) in enumerate(TissueProperties().tissue_names) if 'skin' in s.lower()]
        if tissue_mapping is None:
            skin_ids = skin_ids_TissueProperties
        else:
            # if there is a mapping we have to look up the original ids in the mapping table
            skin_ids = [i for (i, m) in enumerate(tissue_mapping) if m in skin_ids_TissueProperties]

        # set  the values of the skin_ids to one
        model_binary = np.zeros(model.shape)
        for i in skin_ids:
            model_binary[np.nonzero(model == i)] = 1

        outer_shape = {'front': np.sum(model_binary, axis=0),
                       'right': np.sum(model_binary, axis=1)}

        return outer_shape

    @staticmethod
    def calculate_trunk_model(voxel_model: VoxelModel, model_type: str, z_start: int, z_end: int)\
            -> Tuple[np.ndarray, Dict]:
        """
        Calculate the trunk model
        the slice z_end will not be included in the final model
        :param voxel_model:      the voxel model that is converted to a trunk only model
        :param model_type:      the model_type that is used as basis for the conversion
        :param z_start:         start slice
        :param z_end:           end slice (not included)

        """

        logging.info("Calculate the trunk model..")
        model_trunk = voxel_model.models[model_type][:, :, z_start:z_end]
        model_trunk = voxel_model.remove_arms(model_trunk)

        mask_trunk = {'x': range(0, model_trunk.shape[0]),
                      'y': range(0, model_trunk.shape[1]),
                      'z': range(z_start, z_end)}

        return model_trunk, mask_trunk

    def read_tissue_mapping(self, tissue_mapping_file: str):
        """
        Read in the tissue mapping text file. The file has to be formatted the following way:
            ID,Tissue Name, Assigned Tissue Name
            0, ExternalAir, ExternalAir
            250, Some Fat, Fat

        The assigned tissue name has to have the same spelling has the tissue in TissueProperties.

        :param tissue_mapping_file: Filename of the tissue mapping text file
        :return:
        """

        # empty temp list for storing all tissues in the file
        # each entry will be a tuple: (ID, Tissue Name, Assigned Tissue Name)
        tissues_temp = []
        # store the maximum tissue id
        max_t_id = 0
        with open(tissue_mapping_file) as datafile:
            for row in datafile:
                try:
                    (t_id, name, assigned_tissue) = row.split(',')
                except ValueError as e:
                    logging.error("ValueError in row: %s" % row)

                if t_id.isdigit():
                    t_id = int(t_id)
                    if t_id > max_t_id:
                        max_t_id = t_id
                    name = name.strip()
                    assigned_tissue = assigned_tissue.strip()
                    tissues_temp.append((t_id, name, assigned_tissue))

        # look up the assigned_tissues in the TissueProperties.
        tp = TissueProperties()
        self.tissue_names = ['' for x in range(max_t_id + 1)]
        self.tissue_mapping = np.zeros(max_t_id + 1, dtype=int)
        for (t_id, name, assigned_tissue) in tissues_temp:
            self.tissue_names[t_id] = name
            self.tissue_mapping[t_id] = tp.get_id_for_name(assigned_tissue).astype(int)

    def load_avw_file(self, avw_file: str):
        """
        Read in the contents of an AnalyzeAVW file, as they are distributed by the Helmholtz Zentrum München.

        :param avw_file:
        :return:
        """

        logging.info("Read the data from an AVW file")
        # store the offset of the voxel data from the header
        voxel_offset = 0
        with open(avw_file, 'r') as f:

            if self.show_progress_bar:
                pb.progress_bar.max_value = len(f)

            for (line_number, line) in enumerate(f):

                if self.show_progress_bar:
                    pb.progress_bar.update(line_number + 1)
                # Read the header data until the line 'EndInformation' is reached
                if 'EndInformation' in line:
                    break

                # first line has to contain 'AVW_ImageFile'
                if line_number == 0:
                    if 'AVW_ImageFile' not in line:
                        raise TypeError('%s is not in AVW format' % avw_file)
                    else:
                        # read the offset for the beginning of the binary voxel data
                        voxel_offset = int(line.split()[2])

                # All the other lines contain information on the voxel data
                if '=' in line:  # there is a name=value pair in this line
                    line_splt = line.split('=')
                    name = line_splt[0].strip()
                    value = line_splt[1].strip()
                    if value.isnumeric():
                        if value.isdigit():
                            value = int(value)
                        else:
                            value = float(value)
                    else:
                        value = value.strip('"')

                    try:
                        self.data[name] = value
                    except KeyError:
                        logging.error('Failed to set avw_data[%s] to %s' % (name, value))

            if self.show_progress_bar:
                pb.progress_bar.finish()

        if voxel_offset != 0:
            # Read in the voxel data starting at voxel_offset
            with open(avw_file, 'rb') as f:
                f.seek(voxel_offset)
                data_size = self.data['Height'] * self.data['Width'] * self.data['Depth']
                shape = (self.data['Width'], self.data['Height'], self.data['Depth'])
                self.data['image'] = np.reshape(np.fromfile(avw_file, dtype=np.uint8, count=data_size),
                                                newshape=shape,
                                                order='F')
                # the voxel data needs to be transposed to fit to our coordinate system
                self.data['image'] = np.transpose(self.data['image'], (1, 0, 2))
        else:
            raise FileNotFoundError('Error reading the AVW file')

    def load_from_txt_files(self, txt_files_dir: str):
        """
        Read in the data from txt files as provided by the AustinMan/Woman models

        :param txt_files_dir:
        :return:
        """
        logging.info("Read in the data from txt files..")

        def is_numeric_file(x: str) -> bool:
            # check if x is numeric, excluding the last 4 characters ('.txt')
            try:
                int(x[0:-4])
                return True
            except ValueError:
                return False

        def sorting_key(x: str) -> int:
            return int(x[0:-4])

        file_list = [f for f in listdir(txt_files_dir) if isfile(join(txt_files_dir, f)) and is_numeric_file(f)]
        file_list.sort(key=sorting_key)

        # read in the scaling and the extent of the data
        with open(join(txt_files_dir, 'input.txt'), 'r') as f:
            line = f.readline().split()
            x_extent = int(line[0])
            self.data['scale'] = {'x': float(line[1])*1e3}
            line = f.readline().split()
            y_extent = int(line[0])
            self.data['scale']['y'] = float(line[1])*1e3
            line = f.readline().split()
            z_extent = int(line[0])
            self.data['scale']['z'] = float(line[1])*1e3

        Model_orig = np.zeros([x_extent, y_extent, z_extent], dtype=np.uint8)

        if self.show_progress_bar:
            pb.progress_bar.max_value = len(file_list)
        # read in the data from all the files
        for (k, file) in enumerate(file_list):

            if self.show_progress_bar:
                pb.progress_bar.update(k+1)
            with open(join(txt_files_dir, file)) as datafile:
                data = datafile.read()

            # split the content of each file to single numbers
            data = np.array(data.split())
            # convert to int
            data = data.astype(np.uint8)
            # make it a 2D array again
            data = data.reshape((y_extent, -1))

            Model_orig[:, :, k] = data.transpose()

        if self.show_progress_bar:
            pb.progress_bar.finish()

        self.data['image'] = Model_orig

    @staticmethod
    def cluster_endpoints(vm: VoxelModel, compensation: bool = True) -> VoxelModel:
        """
        This function sets the endpoints (RX locations on the abdomen) and clusters them into 16 clusters.
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
        model_name = vm.name

        # defined ranges for the x- and z-Parameters to get endpoints only at the front (belly)
        x_ranges = {'Alvar': (180, 270),
                    'Alvar_lowres': (180, 270),
                    'AustinWoman_v2.5_2x2x2': (230, 350),
                    'AustinWoman_v2.5_1x1x1': (230, 350),
                    'AustinMan_v2.5_2x2x2': (250, 350),
                    'AustinMan_v2.6_1x1x1': (250, 350),
                    'Irene': (150, 200),
                    'Donna': (320, 400),
                    'Golem': (220, 350),
                    'Helga': (300, 400),
                    'Hanako': (180, 250),
                    'Taro': (180, 300),
                    'VisibleHuman': (300, 450),
                    'Katja': (175, 300),
                    'Frank': (200, 380)}

        z_ranges = {'Alvar': (920, 1230),
                    'Alvar_lowres': (920, 1230),
                    'AustinWoman_v2.5_2x2x2': (900, 1250),
                    'AustinWoman_v2.5_1x1x1': (900, 1220),
                    'AustinMan_v2.5_2x2x2': (1000, 1350),
                    'AustinMan_v2.6_1x1x1': (1000, 1350),
                    'Irene': (950, 1280),
                    'Donna': (1000, 1300),
                    'Golem': (900, 1270),
                    'Hanako': (790, 1128),
                    'Taro': (764, 1020),
                    'Helga': (330, 685),
                    'VisibleHuman': (380, 750),
                    'Katja': (910, 1220),
                    'Frank': (100, 400)}

        # navel for the models set by hand set by hand
        navel = {'Alvar': np.array([246, 335, 1088]),
                 'Alvar_lowres': np.array([246, 335, 1088]),
                 'AustinWoman_v2.5_2x2x2': np.array([304, 355, 1048.]),
                 'AustinWoman_v2.5_1x1x1': np.array([304, 355, 1048.]),
                 'AustinMan_v2.5_2x2x2': np.array([308,  339, 1132]),
                 'AustinMan_v2.6_1x1x1': np.array([308,  339, 1132]),
                 'Irene': np.array([180, 247, 1095.]),
                 'Donna': np.array([375, 210, 1126.]),
                 'Golem': np.array([268, 266, 1066]),
                 'Helga': np.array([363.58, 238.14, 490.]),
                 'Hanako': np.array([224, 309, 946]),
                 'Taro': np.array([218, 315, 860]),
                 'VisibleHuman': np.array([385, 234, 520.]),
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
        endpoints_sorted = sorted(e, key=lambda e: (-e.get_z(), e.get_y()))
        endpoints_sorted_array = np.array(endpoints_sorted)

        # Cluster Mapping
        # split the endpoints in four regions
        clusterA, clusterB, clusterC, clusterD, cluster_map_1st = VoxelModelImporter.clustering(endpoints_sorted_array,
                                                                                                compensation=compensation)

        # subclustering: split the above clusters again into four regions
        clusterA1, clusterA2, clusterA3, clusterA4, cluster_mapA = VoxelModelImporter.clustering(clusterA,
                                                                                                 compensation=compensation)
        clusterB1, clusterB2, clusterB3, clusterB4, cluster_mapB = VoxelModelImporter.clustering(clusterB,
                                                                                                 compensation=compensation)
        clusterC1, clusterC2, clusterC3, clusterC4, cluster_mapC = VoxelModelImporter.clustering(clusterC,
                                                                                                 compensation=compensation)
        clusterD1, clusterD2, clusterD3, clusterD4, cluster_mapD = VoxelModelImporter.clustering(clusterD,
                                                                                                 compensation=compensation)

        if model_name == 'Alvar':
            # some manual manipulation for Alvar is needed.
            cluster_mapC[18] = [0.]
            clusterC1 = np.vstack((clusterC1[0:8], clusterC3[0], clusterC1[8::]))
            clusterC3 = clusterC3[1::]

        # For each endpoint find the subcluster and set up a cluster map for the 16 clusters
        # cluster_map: defines overall mapping
        # array of shape [len(endpoints),2] which indicates for each endpoint the cluster to which it belongs
        # A1 | A2 | B1 | B2 |
        # A3 | A4 | B3 | B4 |
        # C1 | C2 | D1 | D2 |
        # C3 | C4 | D3 | D4 |
        # first number indicates cluster A,B,C,D; second number the subcluster 1,2,3,4

        cluster_map = np.concatenate((cluster_map_1st, np.zeros((len(cluster_map_1st), 1))), axis=1)
        for (e, i) in zip(endpoints_sorted_array, range(len(endpoints_sorted))):
            e_list = e.tolist()
            if e_list in clusterA.tolist():
                clusterA_list = clusterA.tolist()
                index = clusterA_list.index(e_list)
                cluster_map[i, 1] = cluster_mapA[index, 0]
            if e_list in clusterB.tolist():
                clusterB_list = clusterB.tolist()
                index = clusterB_list.index(e_list)
                cluster_map[i, 1] = cluster_mapB[index, 0]
            if e_list in clusterC.tolist():
                clusterC_list = clusterC.tolist()
                index = clusterC_list.index(e_list)
                cluster_map[i, 1] = cluster_mapC[index, 0]
            if e_list in clusterD.tolist():
                clusterD_list = clusterD.tolist()
                index = clusterD_list.index(e_list)
                cluster_map[i, 1] = cluster_mapD[index, 0]

        # add the endpoints and the cluster mapping as variables to the truncated model
        vm.models['trunk'].endpoints_abdomen = endpoints_sorted
        vm.models['trunk'].endpoints_abdomen_clustering = cluster_map

        return vm

    @staticmethod
    def clustering(endpoints: np.array, compensation: bool = True):
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

        if compensation:
            # compensate between A and C in case the endpoints are badly distributed
            diff_A_C = len(clusterA) - len(clusterC)
            if diff_A_C > 0:  # cluster A has more endpoints than cluster C
                last_z = clusterA_array[len(clusterA) - 1][2]  # z-value of last endpoint in A
                lr = 1
                counter = len(clusterA) - 2
                # count the endpoints with the same z-value as the last endpoint (10mm difference are allowed) --> this points
                # lie in a row
                while (clusterA_array[counter][2] < last_z + 10):
                    lr += 1
                    counter -= 1
                if np.abs(diff_A_C - 2 * lr) < diff_A_C:  # compensation is advantageous
                    comp = clusterA_array[len(clusterA_array) - lr: len(clusterA_array)]
                    clusterC_array = np.concatenate((comp, clusterC_array))  # last row from A is concatenated with C
                    clusterA_array = clusterA_array[0:len(clusterA_array) - lr]  # A is shortened
            elif diff_A_C < 0:  # cluster C has more endpoints than cluster A
                first_z = clusterC_array[0][2]  # z-value of first endpoint in C
                fr = 1
                counter = 1
                while (clusterC_array[counter][2] > first_z - 10):
                    fr += 1
                    counter += 1
                if np.abs(diff_A_C + 2 * fr) < np.abs(diff_A_C):  # compensation is advantageous
                    comp = clusterC_array[0: fr]
                    clusterA_array = np.concatenate((clusterA_array, comp))  # the row from C is concatenated with A
                    clusterC_array = clusterC_array[fr:]

            # compensate between B and D
            diff_B_D = len(clusterB) - len(clusterD)
            if diff_B_D > 0:  # cluster B has more endpoints than cluster D
                last_z = clusterB_array[len(clusterB) - 1][2]  # z-value of last endpoint in B
                lr = 1
                counter = len(clusterB) - 2
                # count the endpoints with the same z-value as the last endpoint (10mm difference are allowed) --> this points
                # lie in a row
                while (clusterB_array[counter][2] < last_z + 10):
                    lr += 1
                    counter -= 1
                if np.abs(diff_B_D - lr) < diff_B_D:  # compensation is advantageous
                    comp = clusterB_array[len(clusterB_array) - lr: len(clusterB_array)]
                    clusterD_array = np.concatenate((comp, clusterD_array))
                    clusterB_array = clusterB_array[0:len(clusterB_array) - lr]
            elif diff_B_D < 0:  # cluster D has more endpoints than cluster B
                first_z = clusterD_array[0][2]  # z-value of first endpoint in D
                fr = 1
                counter = 1
                while (clusterD_array[counter][2] > first_z - 10):
                    fr += 1
                    counter += 1
                if np.abs(diff_B_D + fr) < np.abs(diff_B_D):  # compensation is advantageous
                    comp = clusterD_array[0: fr]
                    clusterB_array = np.concatenate((clusterB_array, comp))  # the row from D is concatenated with B
                    clusterD_array = clusterD_array[fr:]

        # set up the cluster mapper
        for (e, i) in zip(endpoints, range(len(endpoints))):
            e = e.tolist()
            if e in clusterA_array.tolist():
                cluster_mapper[i, :] = 0
            elif e in clusterB_array.tolist():
                cluster_mapper[i, :] = 1
            elif e in clusterC_array.tolist():
                cluster_mapper[i, :] = 2
            elif e in clusterD_array.tolist():
                cluster_mapper[i, :] = 3

        return clusterA_array, clusterB_array, clusterC_array, clusterD_array, cluster_mapper

    @staticmethod
    def determine_physiological_properties(vm: VoxelModel, navel: Coordinate,
                                           weight: float, height: float, age: float, sex: str) -> VoxelModel:
        """
        Determine the physiological properties of the model, given weight in kg, height in mm, age,
        and sex (male or female)
        :param vm:
        :return:
        """
        bmi = weight / (height/1e3)**2

        # Compute the waist circumference
        # extract the slice at height of the navel from the trunk model
        navel_index = vm.models['trunk'].coordinate_to_index(navel, vm.scaling)
        navel_slice = vm.models['trunk'][:, :, navel_index[2]].astype(np.uint8) > 0
        navel_slice = navel_slice.astype('uint8')
        # excluded in slice_binary.
        kernel = np.ones((3, 3), np.uint8)
        navel_slice_dil = cv.dilate(navel_slice, kernel)

        circumference = navel_slice_dil - navel_slice

        first = True
        waist_cir = 0
        x_incr = vm.scaling.x
        y_incr = vm.scaling.y
        xy_incr = np.sqrt(vm.scaling.x**2 + vm.scaling.y**2)

        while np.sum(circumference > 0) > 0:
            if first:
                first = False
                c_temp = np.transpose(np.nonzero(circumference))
                cur_pos = tuple(c_temp[0, :])
                # always set the active position to 0 already -> it has been counted
                circumference[cur_pos] = 0

            # get the 3x3 neighborhood of the cur position in the image
            p = circumference[cur_pos[0]-1:cur_pos[0]+2, cur_pos[1]-1:cur_pos[1]+2]
            # first check for connection in x direction
            # check left of cur_pos
            if p[1, 0] == 1:
                waist_cir += x_incr
                cur_pos = (cur_pos[0], cur_pos[1] - 1)
            # check right of cur_pos
            elif p[1, 2] == 1:
                waist_cir += x_incr
                cur_pos = (cur_pos[0], cur_pos[1] + 1)
            # check above cur_pos
            elif p[0, 1] == 1:
                waist_cir += y_incr
                cur_pos = (cur_pos[0] - 1, cur_pos[1])
            # check below cur_pos
            elif p[2, 1] == 1:
                waist_cir += y_incr
                cur_pos = (cur_pos[0] + 1, cur_pos[1])
            # check for above left
            elif p[0, 0] == 1:
                waist_cir += xy_incr
                cur_pos = (cur_pos[0] - 1, cur_pos[1] - 1)
            # check for above right
            elif p[0, 2] == 1:
                waist_cir += xy_incr
                cur_pos = (cur_pos[0] - 1, cur_pos[1] + 1)
            # check for below right
            elif p[2, 2] == 1:
                waist_cir += xy_incr
                cur_pos = (cur_pos[0] + 1, cur_pos[1] + 1)
            # check for below left
            elif p[2, 0] == 1:
                waist_cir += xy_incr
                cur_pos = (cur_pos[0] + 1, cur_pos[1] - 1)
            # no adjacent pixel is nonzero => maybe the circumference line is broken -> start new
            else:
                first = True

            # last set cur_pos to zero, such that it is not detected anymore.
            circumference[cur_pos] = 0

        # Waist to height ratio
        wthr = waist_cir / height

        # Fat mass in the abdomen
        fat_id, muscle_id = tuple(TissueProperties().get_id_for_name(["Fat", "Muscle"]))
        fat_density = 0.9 * 1e-3
        data_trunk = vm.models['trunk'][:, :, :]
        n_fat_voxels_trunk = data_trunk[data_trunk == fat_id].size

        fat_mass_trunk = np.prod(np.array(vm.scaling)) * n_fat_voxels_trunk * fat_density

        # Muscle mass in abdomen
        muscle_density = 1.05 * 1e-3
        n_muscle_voxels_trunk = data_trunk[data_trunk == muscle_id].size

        muscle_mass_trunk = np.prod(np.array(vm.scaling)) * n_muscle_voxels_trunk * muscle_density

        muscle_to_fat_ratio = muscle_mass_trunk / fat_mass_trunk

        props = {'sex': sex,
                 'age': age,
                 'height': height,
                 'weight': weight,
                 'bmi': bmi,
                 'waist_circumference_mm': waist_cir,
                 'waist_to_height_ratio': wthr,
                 'abdominal_muscle_to_fat_ratio': muscle_to_fat_ratio,
                 'navel': Coordinate(navel)}

        vm.physiological_properties = PhysiologicalProperties(**props)

        return vm