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


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax, origin: np.ndarray=None, radius: float=None):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Credit: https://stackoverflow.com/a/31364297
    Input
      :param ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    if origin is None:
        origin = np.mean(limits, axis=1)
    if radius is None:
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    set_axes_radius(ax, origin, radius)


class ModelNames:
    # A list of all available VoxelModels (all the ones for which an import script is available)
    AustinMan2 = AustinMan_v25_2mm = 'AustinMan_v2.5_2x2x2'
    AustinMan1 = AustinMan_v26_1mm = 'AustinMan_v2.6_1x1x1'
    AustinWoman1 = AustinWoman_v25_1mm = 'AustinWoman_v2.5_1x1x1'
    AustinWoman2 = AustinWoman_v25_2mm = 'AustinWoman_v2.5_2x2x2'
    Donna = 'Donna'
    Frank = 'Frank'
    Golem = 'Golem'
    Helga = 'Helga'
    Irene = 'Irene'
    Katja = 'Katja'
    VisibleHuman = 'VisibleHuman'
    Alvar = 'Alvar'
    Taro = 'Taro'
    Hanako = 'Hanako'

    # All models that are commonly used for simulations
    all = [Alvar,
           AustinMan_v26_1mm,
           AustinMan_v25_2mm,
           AustinWoman_v25_1mm,
           AustinWoman_v25_2mm,
           Donna,
           Golem,
           Hanako,
           Helga,
           Irene,
           Taro,
           VisibleHuman]

    def __getitem__(self, item):
        """
        Use getitem to generate a list of phantoms
        :param item:
        :return:
        """
        if isinstance(item, str):
            if item == 'all':
                return getattr(self, item)
            else:
                return [getattr(self, item)]
        else:
            return [getattr(self, i) for i in item]
