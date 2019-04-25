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
