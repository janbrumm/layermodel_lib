# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import random
import warnings
import logging
import numpy as np
import gzip
import pickle  # cPickle is used in python3 by default, if available: https://stackoverflow.com/a/19191885
try:
    import cv2 as cv
except ModuleNotFoundError as e:
    warnings.warn("Module opencv-python (cv2) not found.")

import random
import matplotlib as matplt
import matplotlib.pyplot as plt
import scipy.spatial.distance as ssdist

from os import listdir, makedirs, getcwd
from os.path import join, isfile, dirname, exists
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, Optional, List, Tuple, Union, Any, Callable
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap
from matplotlib.collections import PatchCollection
from operator import attrgetter
from scipy.spatial.distance import pdist, squareform
from copy import deepcopy

from LayerModel_lib.general import ProgressBarConfig as ProgBar
from LayerModel_lib.coordinate import Coordinate
from LayerModel_lib.tissue_properties import TissueProperties
from LayerModel_lib.voxelmodel_data import VoxelModelData
from LayerModel_lib.general import set_axes_equal
from LayerModel_lib.general import ModelNames


class VoxelModel:
    """
    A class to store the voxel models from different sources. It has the following attributes:

        working_directory: Has to be set before first usage. Determines where to look for .VoxelModel files.
        name:   The name of the voxel model. If the VoxelModel is saved it is stored as name.VoxelModel in
                the working_directory
        description: A more detailed description of the model.
        models:     A dictionary containing the actual 3D data. Each model usually has an entry
                        -'original' for the original imported model (with the original tissue IDs).
                        -'complete' the same as 'original' but with the tissue IDs mapped to the ones used in our
                                   work.
                        -'trunk' a truncated version of 'complete' containing only the trunk of the model, usually
                                 without the arms.
        scaling: The scaling of the 3D voxel data in x, y, and z direction. I.e. the size of one voxel.
        physiological_properties: Some physiological properties of the respective model, as they are available.

    """

    # Initially set an empty working directory
    working_directory = ""

    @property
    def path(self) -> str:
        """
        Construct the full path to save the VoxelModel to or load it from
        :return:
        """
        if self.working_directory == "":
            self.working_directory = getcwd()
            logging.warning("No working directory set. It was set to \n %s" % self.working_directory)

        return self.working_directory

    @property
    def filename(self) -> str:
        """
        Define the filename of the VoxelModel
        :return:
        """
        return self.name + '.VoxelModel'

    def load_model(self):
        """ Load a voxel model from a .VoxelModel file"""
        filename = join(self.path, self.filename)

        # store the currently set working_directory
        new_work_dir = self.working_directory

        # check if the file exists
        try:
            f = gzip.open(filename, "rb")
            data = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError()

        # import the loaded data into the same instance of this class
        self.__dict__.clear()
        self.__dict__.update(data)
        # reset the working directory to the value that was set before the data was loaded.
        # As the stored absolute path will most likely be different on different machines
        self.working_directory = new_work_dir

    def __init__(self, model_name: Optional[str] = None):
        """ Generate a VoxelModel Object
        If the argument is empty an empty voxel model is created. Otherwise the voxel model stored in the
        file 'working_directory/model_name.VoxelModel' is loaded.

        :param str model_name: Name of the voxel model to load (without the file ending)

        """

        if model_name is None:
            # initialize the variables
            self.name = 'empty'
            self.description = ''
            self.models = {}
            self.physiological_properties = PhysiologicalProperties()
            self.scaling = Coordinate([1, 1, 1])
            # Load the colormap used for plots
            current_directory = dirname(__file__)
            filename = join(current_directory, 'PhantomColormap.npy')
            self.colormap = matplt.colors.ListedColormap(np.load(filename))
        else:
            self.name = model_name
            self.load_model()

        # can be set to false to activate the progress bar
        self.show_progress_bar = False

    def __bool__(self) -> bool:
        """
        Object evaluates as True only if the model_name is not set to 'empty'
        :return:
        """
        if self.name == 'empty':
            return False
        else:
            return True

    def save_model(self, force_write: bool = False):
        """
        Save the voxel model in the current working_directory.
        """
        filename = join(self.path, self.filename)

        # check if the file exists
        if not force_write and isfile(filename):
            ctrl = input('%s.VoxelModel exists already in\n %s/Phantoms.\n'
                         ' Are you sure you want to overwrite it [y/N]: '
                         % (self.name, self.path))
            if ctrl.lower() == 'y' or ctrl.lower() == 'yes':
                with gzip.open(filename, "wb") as f:
                    pickle.dump(self.__dict__, f)
            else:
                return
        else:
            with gzip.open(filename, "wb") as f:
                pickle.dump(self.__dict__, f)

    def add_voxel_data(self, short_name: str, name: str, model: np.ndarray, outer_shape: Dict,
                       tissue_names: Optional[List[str]] = None,
                       mask: Optional[Dict] = None,
                       tissue_mapping: Optional[np.ndarray] = None):
        """
        Add the model in model to the self.models dictionary of all models derived from the original model.

        :param short_name:  short name used as key for the dictionary
        :param name:        (possibly) longer name that describes the model set
        :param model:       np.ndarray containing the model data
        :param outer_shape: Dictionary containing the 'front' and 'right' view of the contour of the model. Created
                            by VoxelModelImporter.calculate_outer_shape()
        :param tissue_mapping: np.ndarray defining the mapping of tissue IDs in the model
                                to the tissue IDs in TissueProperties
        :param tissue_names: A list of strings containing the tissue names for the integers used in model
        :param mask:         A mask giving the min and max coordinates the voxel data was cut out from the 'original'
                             or 'complete' model
        :return:
        """
        if short_name in self.models:
            logging.error('%s is already saved in the models dict.' % short_name)
        else:
            # force the mapping to be present for the original model
            if short_name == 'original' and tissue_mapping is not None:
                raise TypeError("Error: All but the 'original' model need a tissue_mapping np.ndarray!")

            self.models[short_name] = VoxelModelData(name, model, outer_shape, tissue_names, mask,
                                                     tissue_mapping, scaling=self.scaling)

    def export_to_CST(self, model_type: str, path: str = ""):
        """
        Export the specified model_type to a file format suitable for import in CST Microwave Studio.
        :param path:        Path to the location where the model shall be stored at.
        :param model_type:  type of the model to export.
        :return:
        """
        name = self.name + "_" + model_type
        # create a new subfolder
        relative_path = join(path, name)
        if not exists(relative_path):
            makedirs(relative_path)
        else:
            ctrl = input('the folder %s exists already.\n'
                         ' Are you sure you want to overwrite the CST Model inside it [y/N]: '
                         % relative_path)
            if ctrl.lower() == 'y' or ctrl.lower() == 'yes':
                pass
            else:
                return

        # Save the binary data
        data = self.models[model_type].data
        filename = join(relative_path, name + ".lat")
        with open(filename, 'wb') as f:
            # iterate over all the voxels in the order given in "How to make your own voxel file.pdf" from CST
            for vz in range(data.shape[2] - 1, -1, -1):
                for vy in range(data.shape[1] - 1, -1, -1):
                    for vx in range(data.shape[0]):
                        f.write(data[vx, vy, vz].astype(np.uint8).tobytes())

        # create a .vox file
        (nx, ny, nz) = data.shape
        dx = self.scaling.x
        dy = self.scaling.y
        dz = self.scaling.z

        text = "[Version]\n" \
               "1.0\n\n" \
               "[Material]\n" \
               "//f [MHz]	filename\n" \
               "900 %s_Materials.txt" % name + "\n\n" \
               "[Background]\n" \
               "0\n\n" \
               "[Voxel]\n" \
               "//type	nx	ny	nz	dx[mm]	dy[mm]	dz[mm]	offset	filename\n" \
               "char %d %d %d %f %f %f 0 %s.lat" % (nx, ny, nz, dx, dy, dz, name) + "\n\n" \
               "[Bitmap]\n" \
               "front %s_front.bmp\nside %s_side.bmp" % (name, name)

        filename = join(relative_path, name + ".vox")
        with open(filename, 'w') as f:
            f.write(text)

        # create the xx_Materials.txt
        text = "// Tissue	Num	RelPermittivity	RelPermeability	Conductivity	Rho	K \n" \
               "// 				[S/m]	[kg/m^3]	[W/mK]	[J/gK]	[W/m^3K]	[W/m^3]	\n"

        for (id, tissue) in enumerate(self.models[model_type].tissue_names):
            text += "%s \t\t %d \t1.000000 1.000000 0.000000 0.000000\n" % (tissue, id)

        filename = join(relative_path, name + "_Materials.txt")
        with open(filename, 'w') as f:
            f.write(text)

    """ 
    Here begins the section with functions directly working on the voxel models
    """
    def set_scale(self, scale_x: float, scale_y: float, scale_z: float):
        """
        Set the scale of the model in mm
        :param float scale_x:
        :param float scale_y:
        :param float scale_z:
        """
        self.scaling = Coordinate([scale_x, scale_y, scale_z])

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculates the distance between a and b inside the voxel model considering the scaling of the
        model. Each is a numpy.array giving the indices in each dimension of the model.

        :return:
        :param np.ndarray a:     First index in voxel model
        :param np.ndarray b:     Second index in voxel model
        :rtype: float
        :return: Distance in mm between a and b
        """
        dist = abs(a - b)
        return np.linalg.norm(dist * self.scaling)

    def get_tissue_id(self, model_type: str, coord: Coordinate) -> int:
        """
        Get the tissue ID from the position coord in mm.
        :param model_type: name of the model type to use, e.g. 'complete', 'original', or 'trunk'
        :param coord:   coordinate in mm of the desired tissue value. The coordinate is always given with respect
                        to the "complete" or "original" model to make the coordinates usable for all models.

        :return:
        """
        model = self.models[model_type]
        # use the scaling to determine the nearest element in the voxel model
        # this is by definition of the coordinates in the "complete" or "original" coordinate system
        voxel_coords = Coordinate((coord.T / self.scaling).astype(int))
        # transform the coordinates to the coordinates of the model_type
        voxel_coords.x -= model.mask['x'].start
        voxel_coords.y -= model.mask['y'].start
        voxel_coords.z -= model.mask['z'].start

        return model[voxel_coords.x, voxel_coords.y, voxel_coords.z]

    def get_tissue_name(self, model_type: str, coord: Coordinate) -> str:
        """
        Get the tissue name from the position coord in mm.
        :param model_type: name of the model type to use, e.g. 'complete', 'original', or 'trunk'
        :param coord:  coordinate in mm of the desired tissue value.

        :return:
        """
        tissue_id = self.get_tissue_id(model_type, coord)
        return self.models[model_type].tissue_names[tissue_id]

    def tissue_finding_3D(self, startpoint: Coordinate, endpoint: Coordinate, *,
                          return_load_tissue: bool = False, model_type: str = 'trunk') -> Dict:
        """
        Returns all the tissues and the corresponding thickness that can be found
        on the line between startpoint and endpoint in the VoxelModel.


        :param numpy.ndarray startpoint:       coordinate of the startpoint
        :param numpy.ndarray endpoint:         coordinate of the endpoint
        :param bool return_load_tissue:  if this is set to true, additionally the tissue id of the endpoint_d location
                                         is returned
        :param str model_type:          The type of the model that is used for the calculation. Default is 'trunk',
                                        i.e. the model without the arms.
        :rtype: Dict
        :return:
                tissue_array            a vector of all tissues found
                depth_array             the corresponding thicknesses in m
                endpoint                the endpoint coordinate determined by the algorithm. That is the point
                                        that is directly on the skin surface in the direction of
                                        the line connecting start and endpoint given to the
                                        function. Hence, this value can be different to the value
                                        passed to the function above!!
                tissue_load             the tissue id at the endpoint
        """

        # use the desired model
        model = self.models[model_type]

        # Calculate the indices in model the startpoint and endpoint correspond to
        # startpoint_d and endpoint_d give the coordinates in fractional indices to the model. Hence, startpoint_d
        # is basically a scaled version of startpoint.
        startpoint_d = np.array(self.models[model_type].coordinate_to_index(startpoint, self.scaling))
        endpoint_d = np.array(self.models[model_type].coordinate_to_index(endpoint, self.scaling))

        # Make sure start and endpoint are both row vectors
        startpoint_d.shape = (1, -1)
        endpoint_d.shape = (1, -1)

        startpoint_index = startpoint_d.astype(int)

        # -- Define the line connecting endpoint_d and startpoint_d
        # y = m*x + b
        line_direction = endpoint_d - startpoint_d
        # check if there are any coordinates between start and end.
        # That there are none happens only if the startpoint is the same as the
        # endpoint, which only occurs for the source impedance calculation very close to the edge.
        if np.linalg.norm(line_direction) == 0:
            source_tissue = model[startpoint_index[:, 0], startpoint_index[:, 1], startpoint_index[:, 2]]
            load_tissue = source_tissue
            return {'tissue_array': np.array([]), 'depth_array': np.array([]), 'endpoint': endpoint,
                    'source_tissue': source_tissue, 'load_tissue': load_tissue}
        else:
            # We have to make sure that there are enough points on the line per voxel.
            # Otherwise, tissues could be missed.
            # Therefore, include a safety margin of 10 times more pixel than the distance between start and end
            safety_margin = 10

            alpha = np.arange(0, 1, (1/np.linalg.norm(line_direction)/safety_margin))
            # Determine all (x,y,z) coordinates on the line
            x = startpoint_d[0, 0] + alpha * line_direction[0, 0]
            y = startpoint_d[0, 1] + alpha * line_direction[0, 1]
            z = startpoint_d[0, 2] + alpha * line_direction[0, 2]

            # For the further calculation they all need to be column vectors
            x.shape = (-1, 1)
            y.shape = (-1, 1)
            z.shape = (-1, 1)

            # -- get voxel index from the oversampled line coordinates
            # Round first to 4 decimal places to avoid "strange rounding" when the line passes through pixel corners
            x_round = np.around(np.around(x, 4))
            y_round = np.around(np.around(y, 4))
            z_round = np.around(np.around(z, 4))
            # indices in VoxelModel where the line is running through
            c = np.hstack([x_round, np.hstack([y_round, z_round])])
            # get all the unique voxels the line is running through
            u, indices = np.unique(c, axis=0, return_index=True)
            # np.unique sorts the entries, redo the original sorting
            c_unique = np.array([c[index] for index in sorted(indices)])

            # -- ignore all coordinates outside the range of the voxel model
            x_ignore = np.logical_and(c_unique[:, 0] < model.data.shape[0], c_unique[:, 0] >= 0)
            y_ignore = np.logical_and(c_unique[:, 1] < model.data.shape[1], c_unique[:, 1] >= 0)
            z_ignore = np.logical_and(c_unique[:, 2] < model.data.shape[2], c_unique[:, 2] >= 0)
            c_unique = c_unique[np.logical_and(np.logical_and(x_ignore, y_ignore), z_ignore), :].astype(int)

            # -- get the pixel values on the line
            values = model[c_unique[:, 0], c_unique[:, 1], c_unique[:, 2]]

            # -- the endpoint is located at the first pixel outside the body with a
            # value = 0
            endpoint_index = np.nonzero(values == 0)[0]  # all zero elements
            if endpoint_index.size == 0:  # if the size of all zero elements is 0 there were none.
                endpoint_index = c_unique.shape[0]-1
            else:  # otherwise take the first value
                endpoint_index = endpoint_index[0]

            endpoint_d = c_unique[endpoint_index, :]
            endpoint_d.shape = (1, -1)

            # -- In case that there are multiple entries with 0 (Air) cut them off
            values = values[range(0, endpoint_index+1)]
            c_unique = c_unique[range(0, endpoint_index+1), :]

            # -- get the position of the boundaries between different pixel values
            # A 1 at position n indicates that between element n and n+1 there is a change of pixel value.
            # The last element has to be 0 as there is no pixel value next.
            boundaries = np.hstack([abs(np.diff(values)) > 0, False])

            # -- check if there are any boundaries
            # it may happen for the source impedance calculation that there are none
            if np.any(boundaries):
                # -- Calculate the actual intersection with the border between to adjacent different tissues
                # direction in which the next voxel on the line is
                boundary_direction = c_unique[np.hstack([False, boundaries[0:-1]]), :] - c_unique[boundaries, :]
                """ 
                Explanation of the line above: 
                c_unique[boundaries, :] gives the voxel coordinates that are *before* the tissue change.
                c_unique[np.hstack([False, boundaries[0:-2]]), :] gives all pixel coordinates *after* the tissue change. 
                 => the difference is the direction in which you have to go from
                 c_unique[boundaries, :] to the boundary to the next tissue.
                """

                # -- Determine the plane at all boundaries between differing voxel values
                # Plane normal form: (x-p)n= 0 => x*n=p*n
                # Going half the way from c_unique[boundaries, :] to the next voxel on the line gives exactly a point
                # on the plane between the two tissues.
                plane_base = c_unique[boundaries, :] + 0.5*boundary_direction
                # Normal vector has to be orthogonal to the plane ( in the direction of the tissue change )
                plane_normal = boundary_direction

                # -- Calculate intersection between the line and the plane at the border between the
                # two differing pixels
                # This is just the equation for an intersection between line and plane in 3D.
                # alpha_s is the parameter from the line in parameter form.
                alpha_s = (np.diag(plane_base.dot(plane_normal.T)) - plane_normal[:, 0] * startpoint_d[0, 0]
                           - plane_normal[:, 1] * startpoint_d[0, 1] - plane_normal[:, 2] * startpoint_d[0, 2]) \
                          / (plane_normal[:, 0] * line_direction[0, 0] + plane_normal[:, 1] * line_direction[0, 1]
                             + plane_normal[:, 2] * line_direction[0, 2])
                # alpha_s needs to be column vector
                alpha_s.shape = (-1, 1)
                # Number of entries in alpha_s
                no_alpha = alpha_s.shape[0]
                alpha_s.shape = (-1, 1)
                # -- calculate all intersections
                # Going from the startpoint_d alpha_s times the line_direction to get the
                # intersection coordinate with each intersection.
                intersections = np.tile(startpoint_d, (no_alpha, 1)) + np.tile(alpha_s, (1, 3)) \
                                * np.tile(line_direction, (no_alpha, 1))

                # -- Calculate the distances in each medium
                intersection_coordinates = np.vstack([startpoint_d, intersections, endpoint_d])
                # vector to hold the distance in each medium
                depth_array = np.zeros((intersection_coordinates.shape[0] - 1,))
                # calculate distance between intersections
                for k in range(0, intersection_coordinates.shape[0] - 1):
                    depth_array[k] = self.distance(intersection_coordinates[k, :], intersection_coordinates[k+1, :])
                # stack the tissue ids together with the last value
                tissue_array = np.hstack([values[boundaries], values[-1]])

                # -- delete all zero values at the end that might remain in any of the arrays
                depth_array = depth_array[tissue_array > 0]
                tissue_array = tissue_array[tissue_array > 0]
                depth_array = np.around(depth_array, 4)
            else:
                # if there are no boundaries, we are nearly finished.
                tissue_array = np.array([values[0]])
                depth_array = np.array([np.around(self.distance(startpoint_d, endpoint_d), 4)])

            # scale the endpoint_d back to mm
            endpoint = self.models[model_type].index_to_coordinate(endpoint_d, self.scaling)
            if return_load_tissue:
                load_tissue = model[endpoint_d[:, 0], endpoint_d[:, 1], endpoint_d[:, 2]]
            else:
                load_tissue = False

            source_tissue = model[startpoint_index[:, 0], startpoint_index[:, 1], startpoint_index[:, 2]]

            # convert depth_array to values in m
            depth_array = depth_array / 1e3

            return {'tissue_array': tissue_array, 'depth_array': depth_array, 'endpoint': endpoint,
                    'source_tissue': source_tissue, 'load_tissue': load_tissue}

    def remove_arms(self, model: np.ndarray, z_start: Optional[int] = None, z_end: Optional[int] = None) -> np.ndarray:
        """
        Use this function to remove the arms of the complete voxel model. The algorithm just keeps the largest
        object in each transversal slice of the image. It works on all slices in z-direction from z_start
        to z_end (excluded)

        :param model:      The model (np.ndarray) that is used to remove the arms
        :param z_start:    lowest index of the trunk, which does not contain the legs. This is important, as
                                otherwise one of the legs will be deleted.
        :param z_end: end index of the removal of arms operation
        :return: The truncated VoxelModel as np.ndarray
        """

        if z_start is None:
            z_start = 0
        if z_end is None:
            z_end = model.shape[2]

        tp = TissueProperties()
        skin_wet_index = tp.get_id_for_name('SkinWet')
        skin_dry_index = tp.get_id_for_name('SkinDry')

        # make a copy of the voxel model
        new_model = np.copy(model)

        if self.show_progress_bar:
            ProgBar.progress_bar.max_value = len(range(z_start, z_end))
            ProgBar.progress_bar.start()

        # erase the arms, only keep the trunk
        for imageSlice in range(z_start, z_end):

            if self.show_progress_bar:
                ProgBar.progress_bar.update(imageSlice + 1)

            # save the current slice ..
            slice_int = model[:, :, imageSlice].astype(np.uint8)
            # Convert slice_int to a binary image with only ones where the torso is. The selection is made based on all
            # pixels outside the body (value==0) and the ones having the value of skin. This leads to separation of
            # the different areas of the arms and the torso. The skin will later be included again by a dilation
            # operation on the resulting mask.
            slice_binary = np.logical_not(np.logical_or((slice_int == 0),
                                                        np.logical_or((slice_int == skin_wet_index),
                                                                      (slice_int == skin_dry_index)))).astype(np.uint8)

            # compute the center of the image
            # small bugfix for the VisibleHuman model, which has ExternalAir as tissue type in the middle of the
            # trunk ==> move the center if it is found to be ExternalAir. As otherwise the algorithm does not work.
            found_filling_start = False
            x_image_center = np.round(slice_int.shape[0] / 2).astype(int)
            y_image_center = np.round(slice_int.shape[1] / 2).astype(int)
            while not found_filling_start:
                if slice_int[x_image_center, y_image_center] != 0:
                    found_filling_start = True
                else:
                    x_image_center += 2

            mask = np.zeros(slice_binary.shape + np.array([2, 2]), np.uint8)
            # fill the image from the center
            ret, image, mask, rect = cv.floodFill(slice_binary.copy(), mask, (y_image_center, x_image_center), 10)
            # use the mask of all filled points with the original image to produce the output
            # first dilate the mask with a 5x5 kernel to increase the size again and include the skin, that was
            # excluded in slice_binary.
            kernel = np.ones((5, 5), np.uint8)
            output_image = slice_int * cv.dilate(mask[1:-1, 1:-1], kernel)

            new_model[:, :, imageSlice] = output_image

        if self.show_progress_bar:
            ProgBar.progress_bar.finish()

        return new_model

    def get_random_startpoints(self, model_type: str, tissue_type: Union[int, str],
                               number_of_startpoints: int = 1000, min_distance: Optional[float] = 4)\
            -> List[Coordinate]:
        """
        Determines number_of_startpoints randomly inside the tissue_type for the given voxel model type.
        Each startpoint has a minimum distance min_distance to all other startpoints.

        Due to the random nature of selecting the points it can happen that the final number of startpoints
        is diffferent from call to call, if number_of_startpoints is close to the limit of points that can be obtained
        from that model with the given min_distance.

        :param model_type:                  The model type to use
        :param int tissue_type:             The tissue id where the startpoints should be placed.
        :param int number_of_startpoints:   The total number of startpoints
        :param float min_distance:          Minimum distance between adjacent startpoints in mm
        :return:
        """

        if type(tissue_type) is str:
            tissue_type = self.models[model_type].tissue_names.index(tissue_type)

        # Only entries of the tissue type are True
        model_bin = self.models[model_type][:, :, :] == tissue_type
        # get the coordinates of all the points of that tissue type
        # by using np.transpose the coordinates are grouped by element rather than by dimension
        c_temp = np.transpose(np.nonzero(model_bin))

        # scale the coordinates to mm and add the offset of the possibly truncated model
        x = (c_temp[:, 0] + self.models[model_type].mask['x'].start) * self.scaling.x
        y = (c_temp[:, 1] + self.models[model_type].mask['y'].start) * self.scaling.y
        z = (c_temp[:, 2] + self.models[model_type].mask['z'].start) * self.scaling.z

        coordinates_all = np.c_[x, y, z]

        # ensure a given minimum distance between startpoints by starting with the first entry in the array
        # and add subsequent ones only if they have distance greater than min_distance to all other existing coordinates

        rng = np.random.default_rng()
        remaining_idx = np.arange(coordinates_all.shape[0])
        selected_idx = rng.choice(remaining_idx, size=number_of_startpoints, replace=False)

        coordinates_selected = coordinates_all[selected_idx]

        while True:
            # check if the distance to all coordinates in the list is larger than min_distance
            D = ssdist.squareform(ssdist.pdist(coordinates_selected, 'euclidean'))

            # indices of all entries smaller min_distance
            smaller_idx = np.nonzero(np.logical_and(D > 0, D < min_distance))
            if smaller_idx[0].size > 0:
                # there are values smaller min_distance in the vector -> remove them.
                unique_idx = np.unique(smaller_idx[0])
                number_removed = unique_idx.size
                coordinates_selected = np.delete(coordinates_selected, unique_idx, axis=0)
                # draw number_removed new random startpoints from all indices except the ones that have been used already
                # (these are stored in selected_idx
                remaining_idx = np.setdiff1d(np.arange(coordinates_all.shape[0]), selected_idx)
                if remaining_idx.size > 0:
                    if number_removed > remaining_idx.size:
                        # there are not enough remaining coordinates to draw randomly ->
                        # take the remaining as they are
                        new_idx = remaining_idx
                    else:
                        new_idx = rng.choice(remaining_idx, size=number_removed, replace=False)
                else:
                    # there are no remaining coordinates to draw from
                    break

                selected_idx = np.append(selected_idx, new_idx)
                coordinates_selected = np.append(coordinates_selected, coordinates_all[new_idx], axis=0)
            else:
                break

        return_list = [Coordinate(coordinates_selected[i, :]) for i in range(coordinates_selected.shape[0])]

        if len(return_list) < number_of_startpoints:
            logging.warning(f'There are only {len(return_list)} startpoints with min_distance = {min_distance} mm '
                            f'available ({number_of_startpoints} startpoints requrested).')

        return return_list

    def get_random_endpoints(self, model_type: str, number_of_endpoints: int = 1000,
                             min_distance: Optional[float] = None) -> List[Coordinate]:
        """
        Draw number_of_endpoints endpoints from the model_type and return them as List[Coordinate].
        :param model_type: The model type to use
        :param number_of_endpoints: The total number
        :param min_distance: Minimum distance between adjacent endpoints in mm (not implemented yet)
        :return:
        """
        if hasattr(self.models[model_type], 'endpoints'):
            return random.sample(self.models[model_type].endpoints, number_of_endpoints)
        else:
            logging.error('The model type "%s" in "%s" does not have any endpoints associated with it' %
                          (model_type, self.name))

    def show_3d_model(self, model_type: str,
                      show_endpoints: Optional[bool] = False,
                      endpoint_color: Tuple = (0.9255, 0.4313, 0, 0.8),
                      show_clusters: Optional[bool] = False,
                      colored_endpoint_indices: List[Tuple] = None,
                      colored_surf3d_indices: List[Tuple] = None,
                      default_face_color: Tuple = (1, 1, 1, 0.5),
                      show_surf3d_indices: bool = False,
                      show_endpoint_indices: bool = False,
                      x_min_limit: float = None, x_max_limit: float = None,
                      y_min_limit: float = None, y_max_limit: float = None,
                      z_min_limit: float = None, z_max_limit: float = None,
                      plot_origin: np.ndarray = None, plot_radius: float = None,
                      transformation_vector: Coordinate = None,
                      axes_off = True) -> Tuple:
        """
        Open a plot showing the 3D model of the model_type given.

        :param str model_type: model type that is to be plotted, e.g. 'trunk'
        :param bool show_endpoints: if set to True the endpoint vertices are coloured
        :param endpoint_color: Color used for the endpoints if show_endpoints is True
        :param bool show_clusters: if set to True the 16 clusters are shown with different colours
        :param List[Tuple] colored_endpoint_indices: List of endpoint indices that should be colored. Each list entry
                                                     is a tuple (color, hatch_pattern, list of indices).
                                                     The indices are taken from vm.models['trunk'].endpoints_abdomen.
                                                     This overrides the setting of show_clusters and show_endpoints.
        :param colored_surf3d_indices:  same as 'colored_endpoint_indices' only that indices into
                                        model['trunk'].surface_3d are given.
        :param show_surf3d_indices: Show the index of each surface patch
        :param default_face_color: Default color of the patches given as (r, g, b, alpha) tuple.
        :param x_max_limit: maximum x value to plot
        :param x_min_limit: minimum x value to plot
        :param y_max_limit: maximum y value to plot
        :param y_min_limit: minimum y value to plot
        :param z_max_limit: maximum z value to plot
        :param z_min_limit: minimum z value to plot
        :param plot_origin: origin for setting axis scaling equal -> handed over to set_axes_equal()
        :param plot_radius: radius for setting axis scaling equal -> handed over to set_axes_equal()
        :param transformation_vector: a vector giving the translation applied to all patches before drawing
        :param axes_off: turns axes of the figure off by default
        :return:
        """

        # colours for the different clusters
        cluster_colour = {'[0. 0.]': (0, 0, 0.5451, 0.8), '[0. 1.]': (0, 0, 1, 0.8),
                          '[0. 2.]': (0.2824, 0.4627, 1, 0.8), '[0. 3.]': (0.5294, 0.8078, 1, 0.8),
                          '[1. 0.]': (0, 0.3922, 0, 0.8), '[1. 1.]': (0, 0.7882, 0.3412, 0.8),
                          '[1. 2.]': (0.4, 0.8039, 0, 0.8), '[1. 3.]': (0.498, 1, 0, 0.8),
                          '[2. 0.]': (0.5451, 0.1372, 0.1372, 0.8), '[2. 1.]': (1, 0.251, 0.251, 0.8),
                          '[2. 2.]': (1, 0.6940, 0.0118, 0.8), '[2. 3.]': (1, 0.3804, 0.0118, 0.8),
                          '[3. 0.]': (0.5451, 0.4118, 0.0784, 0.8), '[3. 1.]': (0.9333, 0.7059, 0.1333, 0.8),
                          '[3. 2.]': (1, 0.8431, 0, 0.8), '[3. 3.]': (1, 1, 0, 0.8)
                          }

        # generate a list of all endpoints that are supposed to be colored
        colored_endpoints = []
        if colored_endpoint_indices is not None:
            for (color, hatch, indices) in colored_endpoint_indices:
                endpoints = [e for (index, e) in enumerate(self.models['trunk'].endpoints_abdomen)
                             if index in indices]
                colored_endpoints.append((color, hatch, endpoints))

        # generate a list of all surface_3d elements that are supposed to be colored
        colored_surf3d = []
        if colored_surf3d_indices is not None:
            for (color, hatch, indices) in colored_surf3d_indices:
                endpoints = [Coordinate(e['centroid']) for (index, e) in enumerate(self.models['trunk'].surface_3d)
                             if index in indices]
                colored_surf3d.append((color, hatch, endpoints))


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_aspect('equal')
        ax.set_box_aspect((1, 1, 1))

        # save a list of min. and max. X, Y, Z coordinates
        x_min, y_min, z_min = [1e10] * 3
        x_max, y_max, z_max = [0] * 3

        # plot the surface:
        for (surf3d_idx, s) in enumerate(self.models[model_type].surface_3d):
            verts = s['verts']
            centroid = s['centroid']

            patch_out_of_z_limits = False

            # find min and max data points
            for v in verts:
                if np.max(v[:, 0]) > x_max:
                    x_max = np.max(v[:, 0])
                if np.min(v[:, 0]) < x_min:
                    x_min = np.min(v[:, 0])
                if np.max(v[:, 1]) > y_max:
                    y_max = np.max(v[:, 1])
                if np.min(v[:, 1]) < y_min:
                    y_min = np.min(v[:, 1])
                if np.max(v[:, 2]) > z_max:
                    z_max = np.max(v[:, 2])
                if np.min(v[:, 2]) < z_min:
                    z_min = np.min(v[:, 2])
                if x_max_limit is not None and np.min(v[:, 0]) > x_max_limit:
                    patch_out_of_z_limits = True
                if x_min_limit is not None and np.max(v[:, 0]) < x_min_limit:
                    patch_out_of_z_limits = True
                if y_max_limit is not None and np.min(v[:, 1]) > y_max_limit:
                    patch_out_of_z_limits = True
                if y_min_limit is not None and np.max(v[:, 1]) < y_min_limit:
                    patch_out_of_z_limits = True
                if z_max_limit is not None and np.min(v[:, 2]) > z_max_limit:
                    patch_out_of_z_limits = True
                if z_min_limit is not None and np.max(v[:, 2]) < z_min_limit:
                    patch_out_of_z_limits = True

            if patch_out_of_z_limits:
                continue

            # move all coordinates by the vector given in transformation_vector
            if transformation_vector is not None:
                verts_transformed = []
                for v in verts:
                    verts_transformed.append(v - np.tile(np.array(transformation_vector), (4, 1)))
                verts = verts_transformed

            surf = Poly3DCollection(verts)

            # colour the patches which are endpoints
            if show_endpoints:
                endpoints_abdomen = np.array(self.models['trunk'].endpoints_abdomen)
                if Coordinate(centroid) in self.models['trunk'].endpoints_abdomen:
                    surf.set_facecolor(endpoint_color)

                else:
                    surf.set_facecolor(default_face_color)
            # colour the endpoint-patches depending on their cluster
            elif show_clusters:
                endpoints_abdomen = np.array(self.models['trunk'].endpoints_abdomen)
                if centroid.tolist() in endpoints_abdomen.tolist():
                    center = Coordinate(centroid)
                    index_center = self.models['trunk'].endpoints_abdomen.index(center)
                    cluster = str(self.models['trunk'].endpoints_abdomen_clustering[index_center])
                    surf.set_facecolor(cluster_colour[cluster])
                else:
                    surf.set_facecolor(default_face_color)
            else:
                surf.set_facecolor(default_face_color)

            if colored_endpoint_indices is not None:
                found_one = False
                for (color, hatch, endpoints) in colored_endpoints:
                    if Coordinate(centroid) in endpoints:
                        hatch_temp = surf.get_hatch()
                        hatch_temp = hatch if hatch_temp is None else hatch_temp + hatch
                        surf.set_facecolor(color)
                        surf.set_alpha(1)
                        surf.set_hatch(hatch_temp)
                        logging.debug('set color to %s at endpoint %s' % (str(color), str(Coordinate(centroid))))
                        found_one = True

                if not found_one:
                    surf.set_facecolor(default_face_color)

            if colored_surf3d_indices is not None:
                found = 0
                color_list = []
                for (color, hatch, endpoints) in colored_surf3d:
                    if Coordinate(centroid) in endpoints:
                        hatch_temp = surf.get_hatch()
                        hatch_temp = hatch if hatch_temp is None else hatch_temp + hatch
                        surf.set_facecolor(color)
                        surf.set_alpha(1)
                        surf.set_hatch(hatch_temp)
                        logging.debug('set color to %s at endpoint %s' % (str(color), str(Coordinate(centroid))))
                        found += 1
                        color_list.append(color)

                if found == 2:
                    verts = [verts[0][((1, 0, 3),)], verts[0][1:4]]
                    surf = Poly3DCollection(verts)
                    surf.set_facecolor(color_list)
                elif found > 2:
                    logging.warning("More than 2 colors per patch are currently not supported.")

            index_text = None
            if show_surf3d_indices:
                c = surf.get_facecolor()
                c[0][3] = 0.5
                surf.set_facecolor(c)
                index_text = f'{surf3d_idx}'

            if show_endpoint_indices:
                try:
                    # get the endpoint index of the current patch
                    endpoint_idx = self.models['trunk'].endpoints_abdomen.index(Coordinate(centroid))
                    c = surf.get_facecolor()
                    c[0][3] = 0.5
                    surf.set_facecolor(c)
                    index_text = f'{endpoint_idx}'

                except ValueError:  # if centroid is not in endpoint list a ValueError is raised
                    pass

            surf.set_edgecolor('k')
            ax.add_collection3d(surf)

            # annotate the indices if desired
            if index_text is not None:
                ax.text(centroid[0], centroid[1], centroid[2], index_text, None, horizontalalignment='center',
                        fontsize='x-small')

            ax.plot(np.array([centroid[0]]), np.array([centroid[1]]), np.array([centroid[2]]), '.',
                    color=[1, 1, 1, 0.5])

        # plot boundaries of the complete model
        x_min = self.models[model_type].mask['x'][0] * self.scaling.x
        x_max = self.models[model_type].mask['x'][-1] * self.scaling.x
        y_min = self.models[model_type].mask['y'][0] * self.scaling.y
        y_max = self.models[model_type].mask['y'][-1] * self.scaling.y
        z_min = self.models[model_type].mask['z'][0] * self.scaling.z
        z_max = self.models[model_type].mask['z'][-1] * self.scaling.z

        if transformation_vector is not None:
            x_min = x_min - transformation_vector.x
            x_max = x_max - transformation_vector.x
            y_min = y_min - transformation_vector.y
            y_max = y_max - transformation_vector.y
            z_min = z_min - transformation_vector.z
            z_max = z_max - transformation_vector.z

        ax.set_zlim(z_min, z_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        set_axes_equal(ax, origin=plot_origin, radius=plot_radius)
        ax.elev = 19
        ax.azim = 10
        ax.dist = 6
        if axes_off:
            ax.set_axis_off()
        return fig, ax

    def determine_physical_endpoint_mapping(self):
        """
        Generate a list of list that contains the physical positions of the endpoints.
        The rows are the physical rows of the abdominal endpoints. However, Entries in the i-th column do not necessarily
        align completely.
        Compare with result of vm.plot_abdominal_endpoint_patches(show_endpoint_indices=False)
        The resulting list assigns the left-top most endpoint the index 0, 0. Hence,
        physical_endpoint_mapping[0][0] is the left top mos endpoint when viewed from the front.
        physical_endpoint_mapping[row][col] : column is increasing from left to right (viewed from front),
                                              row is increasing from top to bottom.
        :param vm:
        :return:
        """

        def multisort(xs, specs):
            # sorting for multiple values in a list.
            for key, reverse in reversed(specs):
                xs.sort(key=key, reverse=reverse)

            return xs

        model_type = 'trunk'
        # make sure not to change the original voxel model
        original_endpoint_list = deepcopy(self.models[model_type].endpoints_abdomen)
        surface_3d = deepcopy(self.models[model_type].surface_3d)

        # for some models, the z coordinates of the centroids of some of the patches are slightly deviating from
        # the majority, that needs to be corrected for the algorithm below.
        z_mapping = {ModelNames.Alvar: {944.9767441860465: 945.4883720930233, 975.1627906976745: 975.6744186046512,
                                        1005.3488372093024: 1005.8604651162791, 1065.7209302325582: 1066.2325581395348,
                                        1066.7441860465117: 1066.2325581395348, 1095.906976744186: 1096.418604651163,
                                        1096.9302325581396: 1096.418604651163, 1156.2790697674418: 1156.7906976744187},
                     ModelNames.AustinMan_v25_2mm: {1096: 1094, 1066: 1064, 1036: 1034},
                     ModelNames.AustinMan_v26_1mm: {1010: 1009, 1040: 1039, 1070: 1069, 1100: 1099,
                                                    1101: 1099, 1130: 1129, 1310: 1309},
                     ModelNames.AustinWoman_v25_2mm: {1220: 1224, 1226: 1224, 1228: 1224},
                     ModelNames.AustinWoman_v25_1mm: {905: 904, 935: 934, 937: 934, 995: 994, 1025: 1024, 1055: 1054,
                                                      1085: 1084,
                                                      1175: 1174, 1205: 1204},
                     ModelNames.Donna: {1290: 1300},
                     ModelNames.Golem: {928: 920, 960: 952, 992: 984, 1016: 1024, 1056: 1048, 1088: 1080, 1120: 1112,
                                        1152: 1144, 1216: 1208},
                     ModelNames.Hanako: {802: 804, 892: 894, 952: 954, 982: 984, 1012: 1014, 1076: 1074},
                     ModelNames.Irene: {1020: 1015, 1050: 1045, 1080: 1075, 1135: 1140, 1170: 1165, 1195: 1200,
                                        1260: 1255},
                     ModelNames.Taro: {896: 898},
                     ModelNames.VisibleHuman: {390: 385, 420: 415, 450: 445, 480: 475, 510: 505, 540: 535, 570: 565,
                                               600: 595,
                                               630: 625, 660: 655, 690: 685, 720: 715, 750: 745}}

        # apply the mapping to the centroids of the patches
        # NOTE: This also temporarily changes the coordinates in self.models[model_type].endpoints_abdomen
        #       as all entries of endpoints are the sorted entries of self.models[model_type].endpoints_abdomen.
        if self.name in z_mapping:
            m = z_mapping[self.name]
            for p in surface_3d:
                if p['centroid'][2] in m.keys():
                    # find that in the original_endpoint_list
                    try:
                        endpoint_idx = original_endpoint_list.index(Coordinate(p['centroid']))
                    except ValueError:
                        endpoint_idx = None

                    old_z = p['centroid'][2]
                    p['centroid'][2] = m[old_z]
                    if endpoint_idx is not None:
                        original_endpoint_list[endpoint_idx].z = m[old_z]

        # select all the patches for the endpoints and sort them the same way as the endpoints
        patches_unsorted = [p for p in surface_3d if Coordinate(p['centroid']) in original_endpoint_list]
        patches = multisort(list(patches_unsorted),
                            ((lambda p: p['centroid'][2], True), (lambda p: p['centroid'][1], False)))

        physical_endpoint_mapping = []
        surface_3d_mapping = {}
        previous_centroid = Coordinate([0, 0, 0])
        row_idx = 0
        for idx, p in enumerate(patches):
            centroid = Coordinate(p['centroid'])
            # the patches in patches are not sorted in the same order as the "official" list of endpoints.
            # ==> determine the index of the endpoint
            endpoint_idx = original_endpoint_list.index(Coordinate(p['centroid']))
            for surf_idx, s in enumerate(surface_3d):
                if np.all(s['centroid'] == p['centroid']):
                    break  # surf_idx is the value where they are equal

            if previous_centroid.z != centroid.z:
                # start a new row
                if len(physical_endpoint_mapping) > 0:
                    row_idx += 1
                physical_endpoint_mapping.append([])

            physical_endpoint_mapping[row_idx].append(endpoint_idx)
            surface_3d_mapping[endpoint_idx] = surf_idx  # map the endpoint index to an index in surface_3d
            previous_centroid = centroid

        return physical_endpoint_mapping, surface_3d_mapping

    def generate_antenna_grid(self, min_dist: Union[float, str], debug: bool = False):
        """
        Generate a grid of antennas on the abdominal surface with a given distance rounded to the grid-size of the
        surface of 3cm by 3cm.
        Return a list of all endpoints with possible shifted starting positions

        :param min_dist: given in mm
        :param debug: If debug is True, return a list of list -> this corresponds better to the real placement
                      on the body.
        :return:
        """
        physical_endpoint_mapping, surf_3d_map = self.determine_physical_endpoint_mapping()

        if isinstance(min_dist, str) and min_dist == 'check':
            # place the antennas in a check pattern
            antenna_grid = []
            for last_row in ['odd', 'even']:
                temp = []
                for row_idx, row in enumerate(physical_endpoint_mapping):
                    # take only every second element in the first row
                    if last_row == 'odd':
                        if not debug:
                            temp = temp + row[::2]
                        else:
                            temp.append(row[::2])
                        last_row = 'even'
                    elif last_row == 'even':
                        if not debug:
                            temp = temp + row[1::2]
                        else:
                            temp.append(row[1::2])
                        last_row = 'odd'

                antenna_grid.append(temp)
        else:
            # the size of the discrete grid is 30mm by 30mm.
            # We just need to take every ith row and column:
            ith = np.ceil(min_dist / 30).astype(int)

            antenna_grid = []
            for start_row in range(ith):
                for start_col in range(ith):
                    temp = []
                    for row_idx, row in enumerate(physical_endpoint_mapping):
                        # take only every ith row
                        if (row_idx + start_row) % ith == 0:
                            # take only every ith element in each row
                            if not debug:
                                temp = temp + row[start_col::ith]
                            # if debug is True, return a list of list -> this corresponds better to the real placement
                            # on the body
                            else:
                                temp.append(row[start_col::ith])

                    antenna_grid.append(temp)

        return antenna_grid

    def plot_abdominal_endpoint_patches(self, endpoint_colors: Dict = None,
                                        default_color: Union[Tuple, str] = (1, 1, 1, 1),
                                        show_endpoint_indices: bool = False,
                                        hf: plt.Figure = None, ha: plt.Axes = None,
                                        plotted_patch_list: List = None,
                                        update_patch_colors: bool = False, **kwargs) -> Tuple[plt.Figure, plt.Axes, List]:
        """
        Plot all the abdominal endpoint patches as viewed from the front of the body.

        :param endpoint_colors: Dict containing the colors for each abdominal endpoint.
                                key = endpoint index, value = list of colors. For all endpoints not in the dict, the
                                default_color will be used. The color value may be a tuple or list of colors.
        :param default_color: The default color of the patches
        :param hf  Figure handle to plot the data in
        :param ha  Axes handle to plot the data in
        :return:
        """
        model_type = 'trunk'

        color_list = []
        ymin, zmin = 1e10, 1e10
        ymax, zmax = 0, 0

        def multisort(xs, specs):
            # sorting for multiple values in a list.
            for key, reverse in reversed(specs):
                xs.sort(key=key, reverse=reverse)

            return xs

        # make sure not to change the original voxel model
        original_endpoint_list = deepcopy(self.models[model_type].endpoints_abdomen)
        surface_3d = deepcopy(self.models[model_type].surface_3d)
        # sort the abdominal endpoints and the patches in ascending z coordinate and ascending y coordinate
        endpoints = multisort(list(original_endpoint_list),
                              ((attrgetter('z'), False), (attrgetter('y'), False)))
        # select all the patches for the endpoints and sort them the same way as the endpoints
        patches_unsorted = [p for p in surface_3d if Coordinate(p['centroid']) in endpoints]
        patches = multisort(list(patches_unsorted),
                            ((lambda p: p['centroid'][2], False), (lambda p: p['centroid'][1], False)))

        # for some models, the z coordinates of the centroids of some of the patches are slightly deviating from
        # the majority, that needs to be corrected for the algorithm below.
        z_mapping = {ModelNames.Alvar: {944.9767441860465: 945.4883720930233, 975.1627906976745: 975.6744186046512,
                                        1005.3488372093024: 1005.8604651162791, 1065.7209302325582: 1066.2325581395348,
                                        1066.7441860465117: 1066.2325581395348, 1095.906976744186: 1096.418604651163,
                                        1096.9302325581396: 1096.418604651163, 1156.2790697674418: 1156.7906976744187},
                     ModelNames.AustinMan_v25_2mm: {1096: 1094, 1066: 1064, 1036: 1034},
                     ModelNames.AustinMan_v26_1mm: {1010: 1009, 1040: 1039, 1070: 1069, 1100: 1099,
                                                    1101: 1099, 1130: 1129, 1310: 1309},
                     ModelNames.AustinWoman_v25_2mm: {1220: 1224, 1226: 1224, 1228: 1224},
                     ModelNames.AustinWoman_v25_1mm: {905: 904, 935: 934, 937: 934, 995: 994, 1025: 1024, 1055: 1054,
                                                      1085: 1084,
                                                      1175: 1174, 1205: 1204},
                     ModelNames.Donna: {1290: 1300},
                     ModelNames.Golem: {928: 920, 960: 952, 992: 984, 1016: 1024, 1056: 1048, 1088: 1080, 1120: 1112,
                                        1152: 1144, 1216: 1208},
                     ModelNames.Hanako: {802: 804, 892: 894, 952: 954, 982: 984, 1012: 1014, 1076: 1074},
                     ModelNames.Irene: {1020: 1015, 1050: 1045, 1080: 1075, 1135: 1140, 1170: 1165, 1195: 1200,
                                        1260: 1255},
                     ModelNames.Taro: {896: 898},
                     ModelNames.VisibleHuman: {390: 385, 420: 415, 450: 445, 480: 475, 510: 505, 540: 535, 570: 565,
                                               600: 595,
                                               630: 625, 660: 655, 690: 685, 720: 715, 750: 745}}

        # apply the mapping to the centroids of the patches
        # NOTE: This also temporarily changes the coordinates in self.models[model_type].endpoints_abdomen
        #       as all entries of endpoints are the sorted entries of self.models[model_type].endpoints_abdomen.
        if self.name in z_mapping:
            m = z_mapping[self.name]
            for p, e in zip(patches, endpoints):
                if p['centroid'][2] in m.keys():
                    old_z = p['centroid'][2]
                    p['centroid'][2] = m[old_z]
                    e.z = m[old_z]

        distances = squareform(pdist(np.array(endpoints)))

        if not update_patch_colors or plotted_patch_list is None:
            if hf is None or ha is None:
                hf, ha = plt.subplots()

            if plotted_patch_list is None:
                plotted_patch_list = []

            for idx, p in enumerate(patches):
                verts = p['verts'][0]

                # determine the extent of the plot:
                ymin = np.minimum(np.amin(verts[:, 1]), ymin)
                ymax = np.maximum(np.amax(verts[:, 1]), ymax)
                zmin = np.minimum(np.amin(verts[:, 2]), zmin)
                zmax = np.maximum(np.amax(verts[:, 2]), zmax)

                # the patches in patches are not sorted in the same order as the "official" list of endpoints.
                # ==> determine the index of the endpoint
                endpoint_idx = original_endpoint_list.index(Coordinate(p['centroid']))

                if endpoint_colors is not None and endpoint_idx in endpoint_colors:
                    color = endpoint_colors[endpoint_idx]
                else:
                    color = [default_color]

                # find the six nearest neighbors
                nearest_idx = tuple(np.argsort(distances[idx, :])[1:7])
                # all of them with a difference in z of 0 are on the same line
                a = np.array([[endpoints[i].y - endpoints[idx].y, endpoints[i].z - endpoints[idx].z] for i in nearest_idx])
                line_neighbors = np.where(a[:, 1] == 0)[0]
                # the values of b give the y difference to all the points on the same line
                b = a[line_neighbors, 0]
                # these values are now used to compute the negative and positive extent from the centroid.
                # if there is only one neighboring element on the same line, the extent is taken from the 3D vertices of
                # the original 3D surface.
                try:
                    neg_extent = float(np.max(b[np.where(b < 0)])) / 2
                except ValueError:
                    # no negative extent -> at the left border
                    neg_extent = np.min(verts[:, 1]) - p['centroid'][1]

                try:
                    pos_extent = float(np.min(b[np.where(b > 0)])) / 2
                except ValueError:
                    # no positive extend -> at the right border
                    pos_extent = np.max(verts[:, 1]) - p['centroid'][1]

                # compute width and height and create a rectangle
                width = pos_extent - neg_extent
                height = np.max(verts[:, 2]) - np.min(verts[:, 2])
                lower_left = (p['centroid'][1] + neg_extent, np.min(verts[:, 2]))

                try:
                    n_colors = len(color)
                    if isinstance(color, str):
                        raise TypeError("str color")
                except TypeError:
                    color = [color]
                    n_colors = 1

                if n_colors == 1:
                    r = Rectangle(lower_left, width, height, fill=True)
                    r.set_edgecolor('black')
                    r.set_facecolor(color[0])
                    plotted_patch_list.append(r)
                    color_list.append(color[0])
                elif n_colors == 2:
                    xy1 = np.array(lower_left)
                    xy2 = xy1 + np.array([0, height])
                    xy3 = xy1 + np.array([width, 0])
                    p1 = plt.Polygon(np.vstack((xy1, xy2, xy3)))
                    plotted_patch_list.append(p1)
                    color_list.append(color[0])

                    xy1 = np.array(lower_left + np.array([width, height]))
                    xy2 = xy1 - np.array([width, 0])
                    xy3 = xy1 - np.array([0, height])
                    p2 = plt.Polygon(np.vstack((xy1, xy2, xy3)))
                    plotted_patch_list.append(p2)
                    color_list.append(color[1])
                elif n_colors == 3:
                    xy1 = np.array(lower_left)
                    xy2 = xy1 + np.array([0, height])
                    xy3 = xy1 + np.array([0.5 * width, 0])
                    p1 = plt.Polygon(np.vstack((xy1, xy2, xy3)))
                    plotted_patch_list.append(p1)
                    color_list.append(color[0])

                    xy1 = np.array(lower_left + np.array([width, 0]))
                    xy2 = xy1 - np.array([0.5 * width, 0])
                    xy3 = xy1 + np.array([0, height])
                    p2 = plt.Polygon(np.vstack((xy1, xy2, xy3)))
                    plotted_patch_list.append(p2)
                    color_list.append(color[1])

                    xy1 = np.array(lower_left + np.array([width, height]))
                    xy2 = xy1 - np.array([width, 0])
                    xy3 = xy1 - np.array([0.5 * width, height])
                    p3 = plt.Polygon(np.vstack((xy1, xy2, xy3)))
                    plotted_patch_list.append(p3)
                    color_list.append(color[2])
                elif n_colors == 4:
                    midpoint = lower_left + np.array([0.5 * width, 0.5 * height])

                    xy1 = np.array(lower_left)
                    xy2 = xy1 + np.array([0, height])
                    p1 = plt.Polygon(np.vstack((xy1, xy2, midpoint)))
                    plotted_patch_list.append(p1)
                    color_list.append(color[0])

                    xy1 = np.array(lower_left)
                    xy2 = xy1 + np.array([width, 0])
                    p2 = plt.Polygon(np.vstack((xy1, xy2, midpoint)))
                    plotted_patch_list.append(p2)
                    color_list.append(color[1])

                    xy1 = np.array(lower_left) + np.array([width, 0])
                    xy2 = lower_left + np.array([width, height])
                    p3 = plt.Polygon(np.vstack((xy1, xy2, midpoint)))
                    plotted_patch_list.append(p3)
                    color_list.append(color[2])

                    xy1 = np.array(lower_left) + np.array([0, height])
                    xy2 = lower_left + np.array([width, height])
                    p4 = plt.Polygon(np.vstack((xy1, xy2, midpoint)))
                    plotted_patch_list.append(p4)
                    color_list.append(color[3])
                else:
                    logging.warning(f'{n_colors} different colors per patch not supported')

                # ha.plot(verts[:, 1], verts[:, 2], '-')
                # ha.plot(p['centroid'][1], p['centroid'][2], 'X', color='black')

        if update_patch_colors:
            for idx, p in enumerate(patches):
                # the patches in patches are not sorted in the same order as the "official" list of endpoints.
                # ==> determine the index of the endpoint
                endpoint_idx = original_endpoint_list.index(Coordinate(p['centroid']))

                if endpoint_colors is not None and endpoint_idx in endpoint_colors:
                    color_list.append(endpoint_colors[endpoint_idx])
                else:
                    color_list.append(default_color)

            ha.clear()

        our_cmap = ListedColormap(color_list)
        patches_collection = PatchCollection(plotted_patch_list, cmap=our_cmap, pickradius=1)
        patches_collection.set_edgecolor('black')
        patches_collection.set_array(np.arange(len(plotted_patch_list)))
        ha.add_collection(patches_collection)

        if show_endpoint_indices:
            for idx, p in enumerate(patches):
                endpoint_idx = original_endpoint_list.index(Coordinate(p['centroid']))
                ha.text(p['centroid'][1], p['centroid'][2], f'{endpoint_idx}', None,
                        horizontalalignment='center',
                        fontsize='x-small')

        ha.set_xlim(left=ymin, right=ymax)
        ha.set_ylim(bottom=zmin, top=zmax)
        ha.axis('equal')

        return hf, ha, plotted_patch_list

    def create_3d_model(self, model_type: str, patch_size: Tuple[float, float], patches: Optional[List] = None,
                        coordinates: Optional[np.ndarray] = None):
        """
        Create a 3D discretized surface model of the given model_type. This should be a model that contains only one
        main structure, e.g. a trunk model, the head or a single(!) leg.

        :param model_type: name of the model type to use, e.g. 'trunk'
        :param patch_size:  (width, height) of the desired discretized patches in mm
        :param patches:
        :param coordinates:
        :return:
        """
        if patches is None:
            coordinates, patches = self.determine_patches(model_type, patch_size, coordinates)

        discrete_surface = []

        logging.info("\nCreating the 3D surface from the patches..")
        if self.show_progress_bar:
            ProgBar.progress_bar.max_value = len(patches)
            ProgBar.progress_bar.start()

        # iterate over all patches and determine the centroid and the vertices
        for (i, p) in enumerate(patches):

            if self.show_progress_bar:
                ProgBar.progress_bar.update(i + 1)

            # all coordinates of that patch
            c = coordinates[tuple(p), :]
            # determine upper and lower bound of patch
            z_min = np.min(c[:, 2])
            c_min = c[c[:, 2] == z_min, :]
            z_max = np.max(c[:, 2])
            c_max = c[c[:, 2] == z_max, :]
            # calculate distance between all minimum and maximum points to determine the upper and lower corners of
            # the patch
            d_min = ssdist.cdist(c_min, c_min, 'euclidean')
            min_indices = np.unravel_index(np.argmax(d_min), d_min.shape)
            lower_corners = c_min[min_indices, :]

            d_max = ssdist.cdist(c_max, c_max, 'euclidean')
            max_indices = np.unravel_index(np.argmax(d_max), d_max.shape)
            upper_corners = c_max[max_indices, :]

            # all patches should touch each other, therefore, add half a scaling.z in each direction
            upper_corners += np.tile([0, 0, self.scaling.z/2], (2, 1))
            lower_corners -= np.tile([0, 0, self.scaling.z / 2], (2, 1))

            # determine the correct ordering for a closed patch of the upper and lower corners
            d_corner = ssdist.cdist(upper_corners, lower_corners, 'euclidean')
            # if the first coordinates are closer together we need to swap the lower corners to make a nearly
            # rectangular patch (otherwise the lines will cross)
            if d_corner[0, 0] < d_corner[1, 0]:
                verts = [np.vstack((upper_corners, lower_corners[::-1]))]
            else:
                verts = [np.vstack((upper_corners, lower_corners))]

            # approximately determine the centroid:
            # first the coordinate half way between the left and right upper corner:
            max_dist = np.max(d_min)
            d_lower_half = ssdist.cdist(lower_corners, c_min, 'euclidean')
            # get coordinate that is closest to halfway between the left and right corner
            index_lower_half = np.unravel_index(np.argmin(abs(d_lower_half - max_dist / 2)), d_lower_half.shape)
            lower_half = c_min[index_lower_half[1], :]

            max_dist = np.max(d_max)
            d_upper_half = ssdist.cdist(upper_corners, c_max, 'euclidean')
            # get coordinate that is closest to halfway between the left and right corner
            index_upper_half = np.unravel_index(np.argmin(abs(d_upper_half - max_dist / 2)), d_upper_half.shape)
            upper_half = c_max[index_upper_half[1], :]

            # centroid is approximately on the half way between the two points upper_half and lower_half
            direction = (upper_half - lower_half)
            centroid_approx = lower_half + 0.5*direction
            centroid_approx.shape = (-1, 3)
            # determine the distance to all points in p
            d_centroid = ssdist.cdist(centroid_approx, c, 'euclidean')
            # choose the one with the minimum distance as centroid
            centroid_approx_min_index = np.argmin(d_centroid)
            centroid = c[centroid_approx_min_index]

            discrete_surface.append({'verts': verts, 'centroid': centroid})

        if self.show_progress_bar:
            ProgBar.progress_bar.finish()

        return discrete_surface

    def determine_patches(self, model_type: str,
                          patch_size: Tuple[float, float], coordinates: Optional[np.ndarray] = None)\
            -> Tuple[np.ndarray, List]:
        """
        Map all the coordinates to unique patches of the given size in mm. It is assumed that the coordinates
        are already sorted by nearest neighbor for each slice. Meaning that coordinates start in the lowest slice, then
        follow the trajectory around the body and when the first slice is finished the next coordinate is the
        first in the second slice.

        The algorithm iterates over these coordinates. For each coordinate the already existing patch with the closest
        distance is searched for. The beginning of a patch is always determined by the every_nth_row count, which
        depends on the height of the patch. To speed up this process all already finished patches are stored
        separately in Patches. Whereas, the currently active patches are stored in ActivePatches. The coordinates
        from the current slice are stored in ActiveSlice and are appended to ActivePatches every time a new slice
        starts (this is done for the get_nearest_patch_index() function to work properly).

        :param model_type: type of the model to use
        :param patch_size: Tuple(width, height) of the patch dimensions
        :param coordinates: [Optional] an array containing all the outside coordinates
        :return:
        """

        # These two functions are just needed to find the matchin patches
        def get_nearest_patch_index(patches: List, coordinate: np.ndarray) -> int:
            """
            Determines for the given coordinate the closest patch. An index to the patches list is returned
            :param patches: List of indices to the coordinates array of the outer function
            :param coordinate: the coordinate the is currently investigated
            :return:
            """
            global_min_dist = float('inf')
            k_min = 0

            # the smallest possible distance from one voxel to the next that forces an abortion of the minimum search
            largest_distance_to_next_voxel = self.scaling.z
            for (k, p) in enumerate(patches):  # each patch has a list of indices to the coordinates array
                t = tuple(p)  # tuple of all indices, for fancy indexing with numpy
                # distance of the given coordinate to all existing ones in that patch:
                coordinate.shape = (-1, 3)
                dist = ssdist.cdist(coordinate, coordinates[t, :], 'euclidean')
                # check for the minimum
                dist_min = np.min(dist)
                if dist_min < global_min_dist:
                    global_min_dist = dist_min
                    k_min = k
                # the global minimum cannot be smaller the this value
                if global_min_dist < largest_distance_to_next_voxel:
                    break

            return k_min

        def add_active_slice_to_active_patches():
            """
            Append the coordinates of the last ActiveSlice to the ActivePatches.
            :return:
            """
            for (i, s) in enumerate(active_slice):
                for element in s:
                    active_patches[i].append(element)

        width = patch_size[0]
        height = patch_size[1]
        # from the height determine how many rows in z-direction each patch schould contain
        every_nth_row = round(height/self.scaling.z)

        if coordinates is None:
            # determine all coordinates outside the model, the result is sorted by nearest neighbor criterion in
            # ascending slices
            coordinates = self.determine_outside_coordinates(model_type)

        # minimum z coordinate
        z_min = np.min(coordinates[:, 2])

        # stores all the indices to coordinates belonging to patch i in Patches[i] as tuple
        patches = []

        # stores only the currently needed patches, on every_nth_row all ActivePatches are finished and appended to
        # Patches. This list contains only indices of slices that have been completely added to ActiveSlice before.
        active_patches = []
        # this list hold all indices of the current slice and their mapping to the already mapped points in
        # ActivePatches
        active_slice = []
        current_patch_index = 0
        last_z_index = -1

        # accumulate distance in each z-slice for the distance between the coordinates
        dist = 0

        logging.info("\nMapping the coordinates to patches...")
        if self.show_progress_bar:
            ProgBar.progress_bar.max_value = coordinates.shape[0]
            ProgBar.progress_bar.start()

        # iterate through all coordinates and map them to connected patches of size (width, height)
        for k in range(coordinates.shape[0]):  # coordinates.shape[0]):
            # generate an index in z direction starting from 0
            z_index = np.round((coordinates[k, 2] - z_min)/self.scaling.z)

            # check if a new slice has begun
            if last_z_index != z_index:
                start_of_new_slice = True
                add_active_slice_to_active_patches()
                active_slice = [[] for i in range(len(active_patches))]
            else:
                start_of_new_slice = False

            last_z_index = z_index

            # update the progress bar
            if self.show_progress_bar:
                ProgBar.progress_bar.update(k + 1)

            # on every_nth_row new patches need to generated, otherwise they just need to be filled.
            if np.mod(z_index, every_nth_row) == 0:
                if start_of_new_slice:
                    # append all previously active patches to Patches
                    patches = patches + active_patches
                    # add the very first item in the slice to ActivePatches
                    active_patches = [[k]]
                    current_patch_index = len(active_patches) - 1
                    dist = 0
                else:
                    # in all other cases add the travelled distance and decide whether its a new patch or belongs to
                    # the current patch

                    # increase the distance from the last to the current point
                    dist += np.linalg.norm(coordinates[k, :] - coordinates[k-1, :])

                    # as long as the distance traveled along the coordinates is smaller than the width of one patch,
                    # all coordinates belong to this patch
                    if dist < width:
                        # add the index to the current patch
                        active_patches[current_patch_index].append(k)
                    else:
                        # create a new patch
                        active_patches.append([k])
                        current_patch_index = len(active_patches) - 1
                        dist = 0
            else:
                # this is not every_nth_row -> Hence, we need to find the patch the current coordinate belongs to
                # we have to search in active_patches for the closest points

                current_patch_index = get_nearest_patch_index(active_patches, coordinates[k, :])
                active_slice[current_patch_index].append(k)

        if self.show_progress_bar:
            ProgBar.progress_bar.finish()
        # add all the open active patches
        patches = patches + active_patches

        return coordinates, patches

    def determine_outside_coordinates(self, model_type: str) -> np.ndarray:
        """
        Determine all coordinates that are just one pixel outside the voxel model.

        :param str model_type: name of the model type to use, e.g. 'complete', 'original', or 'trunk'
        :rtype: numpy.ndarray
        :return: A Mx3 numpy.ndarray containing all coordinates outside the voxel model
        """

        model = self.models[model_type].data

        coordinates = np.zeros(shape=(0, 3))

        logging.info("\nDetermine the outside coordinates..")
        if self.show_progress_bar:
            ProgBar.progress_bar.max_value = model.shape[2]
            ProgBar.progress_bar.start()

        for z_index in range(0, model.shape[2]):

            if self.show_progress_bar:
                ProgBar.progress_bar.update(z_index + 1)

            slice_int = model[:, :, z_index].astype(np.uint8)
            # construct a binary image only having ones where the model is
            slice_binary = np.logical_not((slice_int == 0)).astype(np.uint8)
            # use dilation to get all the pixels just by one pixel outside the boundary
            kernel = np.ones((3, 3), np.uint8)
            slice_binary_dil = cv.dilate(slice_binary, kernel)
            slice_binary_outside = np.logical_and(np.logical_not(slice_binary),
                                                  slice_binary_dil)
            # get the coordinates of all the points outside the model
            # by using np.transpose the coordinates are grouped by element rather than by dimension
            c_temp = np.transpose(np.nonzero(slice_binary_outside))

            # scale the coordinates to mm and add the offset of the possibly truncated model
            x = (c_temp[:, 0] + self.models[model_type].mask['x'].start) * self.scaling.x
            y = (c_temp[:, 1] + self.models[model_type].mask['y'].start) * self.scaling.y
            z = (z_index + self.models[model_type].mask['z'].start) * self.scaling.z

            # sort the coordinates by nearest neighbor, such that they all form a continuous path
            # from: https://stackoverflow.com/a/37744549
            points = np.c_[x, y]
            # create a nearest neighbors graph
            clf = NearestNeighbors(2).fit(points)
            g = clf.kneighbors_graph()
            # construct a graph from matrix G
            t = nx.from_scipy_sparse_matrix(g)
            # extract the order to cycle through the graph T
            order = list(nx.dfs_preorder_nodes(t, 0))
            # apply the order to the x and y coordinates
            x = x[order]
            y = y[order]

            # make them 2 dimensional arrays for concatenation
            x.shape = (-1, 1)
            y.shape = (-1, 1)
            # stack everything together in the coordinates array
            coordinates = np.vstack((coordinates, np.hstack((x, y, z*np.ones(shape=(x.size, 1))))))

        if self.show_progress_bar:
            ProgBar.progress_bar.finish()

        return coordinates

    def cleanup_3d_model(self, model_type: str, z_max: Union[int, None],
                         bad_indices: List = None, area_threshold: int = 200, debug_plots: bool = False):
        """
        Clean-up the 3d model. Deletes all surface elements smaller than ca. area_threshold mm^2. Moreover, all
        surface elements with z-coordinate larger than z_max are deleted. Additional artifacts to be deleted can be
        given in bad_indices.

        :param model_type: name of the model type to use, e.g. 'complete', 'original', or 'trunk'
        :param z_max: maximum z coordinate in mm, all surface elements (patches) above will be deleted.
        :param bad_indices: list of indices that should be deleted, e.g. large artifacts (that happens with some models)
        :param area_threshold: Threshold for the minimum area of a surface element. The area is approximated by the
                                cross product of two vertices of the patch. Hence, it is not 100 % accurate, but it
                                seems to work well.
        :param debug_plots: toggle some debug plots.
        :return:
        """
        area_list = []
        surf3d_indices = []
        del_indices = []
        del_endpoints = []

        # make a list of all surface elements to be deleted in del_indices.
        for (i, s) in enumerate(self.models[model_type].surface_3d):
            verts = s['verts'][0]
            # centroid = s['centroid']

            # remove all vertices above a certain height and skip the rest:
            if z_max is not None and np.min(verts[:, 2]) > z_max:
                del_indices.append(i)
                continue

            # if i is a surface that is an artifact but too big to be detected.
            if bad_indices is not None and i in bad_indices:
                del_indices.append(i)
                continue

            # approximate area of patch
            a = verts[0] - verts[1]
            b = verts[2] - verts[1]
            area = np.linalg.norm(np.cross(a, b))
            area_list.append(area)

            if area < area_threshold:
                surf3d_indices.append(('red', None, [i]))
                del_indices.append(i)

        if debug_plots:
            hf, ha = self.show_3d_model(model_type, colored_surf3d_indices=surf3d_indices, z_max_limit=z_max)
            ha.set_title("Endpoints to be deleted in %s" % self.name)

        if del_indices:
            # list of surface patches to be deleted and kept
            surface_3d_temp_del = [s for (i, s) in enumerate(self.models[model_type].surface_3d) if i in del_indices]
            surface_3d_temp = [s for (i, s) in enumerate(self.models[model_type].surface_3d) if i not in del_indices]

            # delete any endpoints on the abdomen that were from a now removed surface element
            for s in surface_3d_temp_del:
                centroid = Coordinate(s['centroid'])

                # are there any endpoints in the list of centroids to be deleted?
                if centroid in self.models[model_type].endpoints_abdomen:
                    del_endpoints.append(centroid)
                    logging.debug("found something @ %s" % str(centroid))

            # determine indices of endpoints to be removed
            endpoint_indies = [self.models[model_type].endpoints_abdomen.index(e) for e in del_endpoints]
            # first remove the entry from the cluster mapping
            self.models[model_type].endpoints_abdomen_clustering = np.delete(self.models[model_type].endpoints_abdomen_clustering,
                                                                             endpoint_indies,
                                                                             axis=0)
            logging.debug("Deleted enpdoints %s from endpoints_abdomen_clustering" % str(endpoint_indies))
            # then from the actual endpoints list
            for idx in endpoint_indies:
                del self.models[model_type].endpoints_abdomen[idx]
                logging.debug("Deleted endpoint %d from endpoints_abdomen" % idx)

            # delete the surface elements from the VoxelModel
            self.models[model_type].surface_3d = surface_3d_temp
            logging.info("Deleted %d of %d surface elements from surface_3d" %
                         (len(surface_3d_temp_del), len(self.models[model_type].surface_3d) + len(surface_3d_temp_del)))

            if debug_plots:
                import matplotlib.pyplot as plt
                # show histogram of the area of all surface elements
                plt.figure()
                plt.hist(area_list, 'auto')
                plt.title("%s" % self.name)

                hf, ha = self.show_3d_model(model_type, show_endpoints=True, z_max_limit=z_max)
                ha.set_title("%s" % self.name)

        else:
            logging.info("No patches to remove for %s" % self.name)

    def plot_slice(self, view_point: str = 'top', slice_number: int = None, model_type: str = 'trunk',
                   filter_func: Callable[[np.ndarray], np.ndarray] = None,
                   add_grid: bool = False, hf: 'plt.Figure' = None, ha: 'plt.Axes' = None, **kwargs) \
            -> Tuple['plt.Figure', 'plt.Axes']:
        """
        Open a plot of a slice of this voxel model. The viewpoint may be 'top', 'front', or 'right'.
        :param view_point: 'top', 'front', or 'right'
        :param slice_number: The index of the slice in the given direction to show
        :param model_type: The model type to use (default: trunk)
        :return:
        """
        import matplotlib.pyplot as plt

        model = self.models[model_type]

        if 'cmap' not in kwargs:
            cmap = self.colormap
        else:
            cmap = kwargs['cmap']

        if hf is None or ha is None:
            hf, ha = plt.subplots()
        # draw the slice according to the view_point
        if view_point == 'top':
            # calculate the extent of the image and scale it according to the scaling of the model
            right = max(model.mask['y']) * self.scaling.y + 0.5*self.scaling.y
            left = model.mask['y'].start * self.scaling.y - 0.5*self.scaling.y
            bottom = max(model.mask['x']) * self.scaling.x + 0.5*self.scaling.x
            top = model.mask['x'].start * self.scaling.x - 0.5*self.scaling.x
            extent = (left, right, bottom, top)

            if slice_number is None:
                slice_number = model.data.shape[2] // 2
            im = model[:, :, int(slice_number)]
            origin = 'upper'

            # invert y axis up->bottom y axis
            if not ha.yaxis_inverted():
                ha.invert_yaxis()

            # add some axes labels for better orientation
            ha.set_xlabel('Y in mm')
            ha.set_ylabel('X in mm')

        elif view_point == 'right':
            # calculate the extent of the image depending and scale it according to the scaling of the model
            right = max(model.mask['x']) * self.scaling.x + 0.5*self.scaling.x
            left = model.mask['x'].start * self.scaling.x - 0.5*self.scaling.x
            top = max(model.mask['z']) * self.scaling.z + 0.5*self.scaling.z
            bottom = model.mask['z'].start * self.scaling.z - 0.5*self.scaling.z
            extent = (left, right, bottom, top)

            if slice_number is None:
                slice_number = model.data.shape[1] // 2
            # the image needs to be rotated by 90° counter clockwise for the correct orientation
            # This is probably due to the view numpy handles the acces to multidimensional arrays.
            im = np.rot90(model[:, int(slice_number), :])
            origin = 'upper'

            # # invert again to get a normal bottom->up y axis
            # if ha.yaxis_inverted():
            #     ha.invert_yaxis()

            # add some axes labels for better orientation
            ha.set_xlabel('X in mm')
            ha.set_ylabel('Z in mm')

        elif view_point == 'front':
            # calculate the extent of the image depending and scale it according to the scaling of the model
            right = max(model.mask['y']) * self.scaling.y + 0.5*self.scaling.y
            left = model.mask['y'].start * self.scaling.y - 0.5*self.scaling.y
            top = max(model.mask['z']) * self.scaling.z + 0.5*self.scaling.z
            bottom = model.mask['z'].start * self.scaling.z - 0.5*self.scaling.z
            extent = (left, right, bottom, top)

            if slice_number is None:
                slice_number = model.data.shape[0] // 2
            # the image needs to be rotated by 90° counter clockwise for the correct orientation
            # This is probably due to the view numpy handles the acces to multidimensional arrays.
            im = np.rot90(model[int(slice_number), :, :])
            origin = 'upper'

            # invert again to get a normal bottom->up y axis
            # if ha.yaxis_inverted():
            #     ha.invert_yaxis()

            # add some axes labels for better orientation
            ha.set_xlabel('Y in mm')
            ha.set_ylabel('Z in mm')

        v_min = model.min_tissue_id
        v_max = model.max_tissue_id
        if filter_func is not None:
            im = filter_func(im)
            
        ha.imshow(im, cmap=cmap, vmin=v_min, vmax=v_max,
                  extent=extent, origin=origin)

        # draw grid lines at the voxel boundaries
        if add_grid:
            if view_point == 'top':
                x_step = self.scaling.y
                x_max_index = model.data.shape[1]
                x_offset = model.mask['y'].start
                y_step = self.scaling.x
                y_max_index = model.data.shape[0]
                y_offset = model.mask['x'].start
            elif view_point == 'front':
                x_step = self.scaling.y
                x_max_index = model.data.shape[1]
                x_offset = model.mask['y'].start
                y_step = self.scaling.z
                y_max_index = model.data.shape[2]
                y_offset = model.mask['z'].start
            elif view_point == 'right':
                x_step = self.scaling.x
                x_max_index = model.data.shape[0]
                x_offset = model.mask['x'].start
                y_step = self.scaling.z
                y_max_index = model.data.shape[2]
                y_offset = model.mask['z'].start

            for x in range(x_max_index):
                x = x + 0.5 + x_offset
                ha.plot(x*x_step*np.array([1, 1]), [bottom, top], color='tab:gray')
            for y in range(y_max_index):
                y = y + 0.5 + y_offset
                ha.plot([left, right], y*y_step*np.array([1, 1]), color='tab:gray')

            ha.set_xlim(left=left, right=right)
            ha.set_ylim(top=top, bottom=bottom)

        return hf, ha

    @staticmethod
    def list_all_voxel_models() -> List[str]:
        """
        Returns a list of all VoxelModels available in the working_directory
        :return:
        """
        path = VoxelModel.working_directory
        file_ending = '.VoxelModel'
        file_list = [f[:-len(file_ending)] for f in listdir(path)
                     if isfile(join(path, f)) and f.endswith(file_ending)]

        return file_list

    @staticmethod
    def change_organ_ids(original_human_model: np.ndarray, organ_lut: np.ndarray) -> np.ndarray:
        """
        change_organ_ids(original_human_model, organ_lut)
        Changes the Organ IDs in original_human_model according to
        the values in organ_lut.The index of the LUT is the original ID which is
        to be replaced by the value at that index, e.g. organ_lut[3] = 22 means
        that in the original data set value 3 is replaced by the value 22.

        :param np.ndarray original_human_model:    is a(m x n x s)-size ndarray
        :param np.ndarray organ_lut:               is a vector (array)
        :rtype: np.ndarray
        """

        # elements in this array are set to true if the value was
        # already changed by the for loop
        changed_organs = np.zeros(original_human_model.shape)

        for i in range(1, organ_lut.size):
            # mask of all elements with value i
            replacement_mask = (original_human_model == i)
            # replace the value at all positions the value is found, but not at locations that have been altered already
            original_human_model[np.logical_and(replacement_mask, np.logical_not(changed_organs))] = organ_lut[i]
            # update the changed_organs
            changed_organs = np.logical_or(changed_organs, replacement_mask)

        human_model = original_human_model

        return human_model


class PhysiologicalProperties:
    def __init__(self, sex: str = None, age: float = None, height: float = None, weight: float = None,
                 navel: Coordinate = None,
                 bmi: float = None, waist_circumference_mm: float = None, waist_to_height_ratio: float = None,
                 abdominal_muscle_to_fat_ratio: float = None):
        """
        Set of physiological properties. They are determined by VoxelModelImporter.determine_physiological_properties()
        while importing the phantoms.

        :param sex: female or male
        :param age: Age in years
        :param height: Height in mm
        :param weight: Weight in kg
        :param bmi: Body mass index
        :param waist_circumference_mm: Waist circumference in mm
        :param waist_to_height_ratio: Waist to height ratio
        :param abdominal_muscle_to_fat_ratio: Ratio of muscle mass to fat mass in the abdomen
        """
        self.sex = sex
        self.age = age
        self.weight = weight
        self.height = height
        self.navel = navel
        self.bmi = bmi
        self.waist_circumference_mm = waist_circumference_mm
        self.waist_to_height_ratio = waist_to_height_ratio
        self.abdominal_muscle_to_fat_ratio = abdominal_muscle_to_fat_ratio
