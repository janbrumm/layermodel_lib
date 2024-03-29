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

from typing import List, Dict, Optional, Tuple, Union
from LayerModel_lib.tissue_properties import TissueProperties
from LayerModel_lib.coordinate import Coordinate


class VoxelModelData:
    """
    This is a class that holds the 3D arrays containing the voxel models.

    name :          Name of the voxel model including short description
    data :          The 3D matrix containing the voxel data. The data should be accessed using the overloaded []
                    operator of the VoxelModelData class to make sure that the tissue_mapping is used (if
                    it is present)
    tissue_names:   the names of the tissues in data
    max_tissue_id:  maximum tissue id used (is needed for plotting with plt.imshow())
    min_tissue_id:  minimum tissue id used (is needed for plotting with plt.imshow(), set to 0)
    mask:           gives the indices in x, y, and z direction that were used to cut out the specific model
                    with respect to the 'original' model (that should exist in all cases):
                    For each coordinate a Range is stored
                    mask = {'x': range(x_min, x_max, x_step),
                            'y': range(y_min, y_max, y_step),
                            'z': range(z_min, z_max, z_step)}
    tissue_mapping: The mapping of the tissue indices from the 'original' model (as imported) compared to the ones
                    listed in TissueProperties. This may only be None for name == 'original' (as no mapping is needed
                    in that case).
    outer_shape     The outer_shape of the voxel model viewed from 'right' or 'front' for use with the
                    OuterShapeViewer of the PhantomViewer GUI. Outer shape is supposed to be a dict with the two keys
                    'front' and 'right'. The values should be np.ndarray of the same shape as data, reduced by one
                    dimension, depending on the view.

    """
    def __init__(self, name: str,
                 data: np.ndarray,
                 outer_shape: Dict,
                 tissue_names: Optional[List[str]]=None,
                 mask: Optional[Dict]=None,
                 tissue_mapping: Optional[np.ndarray]=None,
                 scaling: Optional[Coordinate]=None):

        self.name = name
        self.data = data
        self.outer_shape = outer_shape

        self.min_tissue_id = 0  # this is just stored for completeness
        if tissue_names is None:
            tissue_names = TissueProperties().tissue_names

        self.max_tissue_id = len(tissue_names)
        self.tissue_names = tissue_names

        if mask is None:
            mask = {'x': range(0, data.shape[0]),
                    'y': range(0, data.shape[1]),
                    'z': range(0, data.shape[2])}
        self.mask = mask

        self.tissue_mapping = tissue_mapping
        self.surface_3d = []  # list of vertices and centroids of the 3d surface
        self.scaling = scaling

    def __str__(self) -> str:
        s = self.name + "\n" \
            "Data size: (%d, %d, %d)" % self.data.shape + "\n" \
            + str(self.mask)
        return s

    def __getitem__(self, item):
        """
        overload the [] parameter to dynamically look for a tissue mapping table.
        Returns a view on the data if no mapping is present.
        If a mapping is present a copy of the values in data will be returned (as not to modify the original
        voxel model)
        :param item:
        :return:
        """
        if self.tissue_mapping is None:
            return self.data[item]
        else:
            return self.map_tissue_ids(self.data[item])

    def map_tissue_ids(self, data_set: np.ndarray) -> np.ndarray:
        """
        Map all the tissues in the data_set with the self.tissue_mapping to new values
        :param data_set:
        :return:
        """
        # iterating over all tissue_mapping entries and modifying the data from data_set
        # this way was found to be the fastest (in terms of visible lag in the PhantomViewer by scrolling through
        # slices)
        new_data_set = np.zeros(data_set.shape, dtype=int)
        it = np.nditer(self.tissue_mapping, flags=['f_index'])
        while not it.finished:
            if it[0] != 0:
                new_data_set[data_set == it.index] = it[0].astype(int)
            it.iternext()

        return new_data_set

    def coordinate_to_index(self, c: Coordinate, scaling: Coordinate = None) -> Tuple[int, int, int]:
        """
        Converts a coordinate to the voxel model to an index that can be used with the model_type in this
        VoxelModelData

        :param Coordinate scaling: The scaling of the parent VoxelModel
        :param Coordinate c: A coordinate in mm relative to the complete body of the voxel model
        :return:
        """
        if scaling is None:
            scaling = self.scaling

        # first round the coordinate to the nearest voxel
        index = np.array(np.around(c / scaling).astype(int))
        # remove any possible offset from the model_type
        index -= np.array([self.mask['x'].start,
                           self.mask['y'].start,
                           self.mask['z'].start])

        if np.any(index < 0) or np.any(index > self.data.shape):
            raise IndexError("Coordinate %s would yield the index tuple %s. \n "
                             "This out of bounds of the model with shape %s" % (str(c), index, self.data.shape))

        return tuple(index)

    def coordinate_to_voxelcoordinate(self, c: Coordinate, scaling: Coordinate = None) -> Tuple[int, int, int]:
        """
        Converts a coordinate in the voxel model (in mm) to a coordinate relative to the voxel indices.
        That means fractions of indices are allowed. In contrast coordinate_to_index() returns the
        index of the voxel the given coordinate resides in.

        :param Coordinate scaling: The scaling of the parent VoxelModel
        :param Coordinate c: A coordinate in mm relative to the complete body of the voxel model
        :return:
        A coordinate relative to this VoxelModelData model type in voxel coordinates (fractions of voxels not mm)
        """
        if scaling is None:
            scaling = self.scaling

        # first round the coordinate to the nearest voxel
        index = np.array(c / scaling)
        # remove any possible offset from the model_type
        index -= np.array([self.mask['x'].start,
                           self.mask['y'].start,
                           self.mask['z'].start])

        if np.any(index <= -0.5) or np.any(index > self.data.shape):
            raise IndexError("Coordinate %s would yield the index tuple %s. \n "
                             "This out of bounds of the model with shape %s" % (str(c), index, self.data.shape))

        return tuple(index)

    def index_to_coordinate(self, index: Union[Tuple, np.ndarray], scaling: Coordinate = None) -> Coordinate:
        """
        Converts an index to the model_type to a coordinate in mm.

        :param index: index into the VoxelModelData.data np.array
        :param scaling: scaling of the parent voxel model.
        :return:
        """
        if scaling is None:
            scaling = self.scaling

        # make sure that the index is 1-dimensional if it is an numpy array
        if type(index) is np.ndarray:
            # make a copy such that the original index is not altered
            index = np.copy(index)
            index.shape = (-1,)

        # add any possible offset from the model_type
        index += np.array([self.mask['x'].start,
                           self.mask['y'].start,
                           self.mask['z'].start])

        return Coordinate(index * scaling)

    def map_endpoints_to_surf3d_index(self, endpoints: List) -> List:
        """
        Map the given ccordinates in mm to the index of the nearest discrete surface element of the model

        :param endpoints: A list of coordinates in mm
        :return:
        """
        endpoint_index_list = []
        # map the points to the closest ones on the discrete surface of the visible human
        for e in endpoints:
            min_dist = 1e6
            min_index = 0
            for (index, s) in enumerate(self.surface_3d):
                dist = np.linalg.norm(np.abs(s['centroid'] - e))
                if dist < min_dist:
                    min_index = index
                    min_dist = dist
            logging.info('%s -> %s , d = %2.2f' % (str(self.surface_3d[min_index]['centroid']), str(e), min_dist))
            endpoint_index_list.append(min_index)

        return endpoint_index_list

    def get_surf3d_bbox(self, z_max_limit: float = None) -> Tuple[Coordinate, Coordinate]:
        """
        Get the minimum and maximum x, y, z values of this 3d surface

        :return: (minium coordinates, maximum coordinates)
        """
        # save a list of min. and max. X, Y, Z coordinates
        x_min, y_min, z_min = [1e10] * 3
        x_max, y_max, z_max = [0] * 3

        for (surf3d_idx, s) in enumerate(self.surface_3d):
            verts = s['verts']

            # find min and max data points
            for v in verts:
                # check if the patch is higher than the z limit given
                if z_max_limit is not None and np.any(np.max(v[:, 2]) > z_max_limit):
                    break

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

        max_coord = Coordinate([x_max, y_max, z_max])
        min_coord = Coordinate([x_min, y_min, z_min])

        return min_coord, max_coord
