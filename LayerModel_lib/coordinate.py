# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import numpy as np
from typing import Any


class Coordinate(np.ndarray):
    """
    Class for storing coordinates. This is a subclass of a numpy.ndarray with the additional attributes x, y, and z
    for easier access to the individual coordinates.
    """

    def __new__(cls, data: Any):
        """
        create new np.ndarray: based on example from
        https://docs.scipy.org/doc/numpy/user/basics.subclassing.html#slightly-more-realistic-example-attribute-
        added-to-existing-array
        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(data).view(cls)
        if obj.ndim > 1:
            raise ValueError('A coordinate may have a maximum number of dimensions of 1!')
        if obj.size != 3:
            raise ValueError('A coordinate must have exactly 3 entries.')
        # add the new attributes to the created instance
        obj._x = data[0]
        obj._y = data[1]
        obj._z = data[2]

        return obj

    def __array_finalize__(self, obj):
        """
        Initialize all the attributes of object itself

        :param obj:
        :return:
        """
        if obj is None:
            return

        self._x = obj[0]
        self._y = obj[1]
        self._z = obj[2]

    def get_x(self):
        return self[0]

    def set_x(self, value):
        self._x = value
        self[0] = value

    def get_y(self):
        return self[1]

    def set_y(self, value):
        self._x = value
        self[1] = value

    def get_z(self):
        return self[2]

    def set_z(self, value):
        self._x = value
        self[2] = value

    # x, y, z are accessed through these getters and setters.
    x = property(get_x, set_x)
    y = property(get_y, set_y)
    z = property(get_z, set_z)

    def __eq__(self, other):
        """
        Determine equality of coordinates.
        :param other:
        :return:
        """
        if self.x == other.x and self.y == other.y and self.z == other.z:
            return True
        else:
            return False

    # Additionally, __setitem__ needs to be overridden to set _x, _y, and _z if the array is accessed by self[i]
    def __setitem__(self, key, value):
        if key == 0:
            self._x = value
        elif key == 1:
            self._y = value
        elif key == 2:
            self._z = value

        super().__setitem__(key, value)
