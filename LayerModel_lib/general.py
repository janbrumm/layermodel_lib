# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import logging
import pickle
import numpy as np

from os.path import join, isfile
from typing import Any, Dict
from progressbar import ProgressBar, ETA, Percentage, AnimatedMarker, Bar

DictProxyType = type(object.__dict__)


class ProgressBarConfig:
    """
    Stores the config of the progress bar used for various time consuming computations.
    """
    widget = [AnimatedMarker(markers='◐◓◑◒'), ' ', ETA(), ' ',
              Percentage(), ' ', Bar(marker='=', left='[', right=']', fill='-')]

    progress_bar = ProgressBar(widgets=widget)


def save_file(content: Any, filename: str, path: str):
    """
    Save a single file with prior check if the file exists already.

    :param content: The content that is to be saved
    :param str path: Path to the save the file in
    :param str filename: name of the file
    :return:
    """

    logging.info('Saving file: %s ' % filename)
    path_to_file = join(path, filename)
    if isfile(path_to_file):
        ctrl = input('%s exists already in\n %s.\n'
                     ' Are you sure you want to overwrite it [y/N]: '
                     % (filename, path))
        if ctrl.lower() == 'y' or ctrl.lower() == 'yes':
            with open(path_to_file, "wb") as f:
                pickle.dump(content, f)
        else:
            logging.warning("%s NOT saved.." % filename)
            return
    else:
        with open(path_to_file, "wb") as f:
            pickle.dump(content, f)

    logging.info("File '%s' saved." % filename)


def dict_equal(d1: Dict, d2: Dict) -> bool:
    """
    Check if the two dictionaries have the same keys and that their values are identical.

    :param dict d1:
    :param dict d2:
    :return:
    """

    # iterate over the dict with more keys
    # di is the dictionary to iterate over
    # dj is the one to compare to
    if len(d2) > len(d1):
        di = d2
        dj = d1
    else:
        di = d1
        dj = d2
    for key, value in di.items():
        # check if key is also in d2 and if the value is the same
        if key not in dj.keys():
            return False
        else:
            value_j = dj[key]
            if type(value) is dict and type(value_j) is dict:
                    # if its again a dictionary -> recursion
                    if not dict_equal(value, value_j):
                        return False

            elif type(value) is np.ndarray and type(value_j) is np.ndarray:
                if not np.array_equal(value, value_j):
                    return False

            # check if both are the same type of object
            elif type(value) is not type(value_j):
                return False

            elif value != value_j:
                return False

    return True


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Credit: https://stackoverflow.com/a/31364297
    Input
      :param ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])