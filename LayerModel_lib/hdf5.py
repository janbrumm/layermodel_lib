import h5py
import logging
import json
import numpy as np

from typing import Dict, Any, Union
from os import sep
from datetime import datetime

from LayerModel_lib import Coordinate


class HDF5:

    def __init__(self, file: h5py.File, compressed: bool = True):
        """
        Init the HDF5 object with a h5py.File object

        :param file: h5py.File object of the file to be accessed
        :param compressed: Turn on compression (gzip, level 9) when writing datasets?
        """
        self.file = file
        self.filename = file.filename.split(sep)[-1]  # just the filename without the absolut path
        self.dataset_opts = {}
        if compressed:
            self.dataset_opts['compression'] = 'gzip'
            self.dataset_opts['compression_opts'] = 9
            self.dataset_opts['shuffle'] = True
            self.dataset_opts['fletcher32'] = True

    def update_attributes(self, attrs: Dict, fullpath: str):
        """
        Update the attributes of a given group or dataset. If there is a change in value, the attribute is deleted
        and created new.

        :param attrs: dictionary of all attributes.
        :param fullpath: path to group or dataset in self.file
        :return:
        """
        f = self.file
        for a_key, a_val in attrs.items():
            if isinstance(a_val, str):  # strings have to be converted using np.string_ -> see h5py docs
                a_val = np.string_(a_val)

            if a_key not in f[fullpath].attrs:
                logging.debug(f"<{self.filename}> [{fullpath}]: New attribute '{a_key}' = '{a_val}'.")

            elif a_key in f[fullpath].attrs and np.any(f[fullpath].attrs[a_key] != np.array(a_val)):
                logging.debug(f"<{self.filename}> [{fullpath}]: Update attribute '{a_key}' from "
                              f"'{f[fullpath].attrs[a_key]}' to '{a_val}'.")
                # attrs.modify() works only for values of same shape. Hence, delete old attribute first
                # and create a new one afterwards
                del f[fullpath].attrs[a_key]

            else:
                continue

            # Add the atrribute. If HDF5 can't handle the type convert to string.
            try:
                f[fullpath].attrs.create(a_key, a_val)
            except TypeError:
                a_val = np.string_("JSON=" + json.dumps(a_val, default=json_default_converter))
                logging.warning(f"<{self.filename}> [{fullpath}]: The value of '{a_key}' "
                                f" raised a TypeError and was converted to a JSON string '{a_val}'.")
                f[fullpath].attrs.create(a_key, a_val)
                
    def update_dataset(self, key: str, value: Any, path: str = "",
                       attrs: Union[Dict, None] = None, **kwargs):
        """
        Update the dataset with key at path with the given value. Optionally, attributes to that dataset can
        be given.

        :param key:
        :param value:
        :param path:
        :param attrs:
        :param kwargs: will be passed to create_dataset()
        :return:
        """
        f = self.file
        fullpath = path + '/' + key

        # filtering is only possible for non-scalar datasets
        if np.array(value).size == 1:
            opts = {}
        else:
            opts = self.dataset_opts

        if attrs is not None and \
                'python_dict_stored_as_nested_dataset' in attrs and \
                attrs['python_dict_stored_as_nested_dataset']:
            self._save_data_from_dict(path, key=key, d=value, **opts, **kwargs)
        else:
            # create a new data set if it does not exists
            if key not in f[path + '/']:
                logging.debug(f"<{self.filename}> Dataset [{key}] did not exist in {path + '/'}: New dataset created.")

            # update the dataset if something changed
            elif np.any(f[path + '/' + key][()] != np.array(value)):
                # delete the dataset. A new one will be created below
                del f[path + '/' + key]
                logging.debug(f"<{self.filename}> Dataset [{key}]: Values updated.")

            else:
                return  # nothing changed do not update dataset

            # Create the dataset. If that is not succesful, try to convert a list or tuple using np.string_().
            # Otherwise convert to JSON string.
            try:
                f.create_dataset(fullpath, data=value, **opts, **kwargs)
            except TypeError:

                try:
                    if isinstance(value, list) or isinstance(value, tuple):
                        # try to convert using string_(this should work for lists and tuples) and convert the
                        # entries to numpy compatible byte strings
                        value = np.string_(value)
                        f.create_dataset(fullpath, data=value, **opts, **kwargs)
                        logging.warning(f"<{self.filename}> [{path}]: The value of dataset [{key}] "
                                        f" raised a TypeError and was converted using np.string_() '{value}'.")
                    else:
                        raise TypeError("It's not a list or tuple!")

                except TypeError:
                    # still does not work -> convert to JSON
                    value = np.string_("JSON=" + json.dumps(value, default=json_default_converter))
                    opts = {}  # options have to be empty. A string is a scalar dataset
                    f.create_dataset(fullpath, data=value, **opts, **kwargs)
                    logging.warning(f"<{self.filename}> [{path}]: The value of dataset [{key}] "
                                    f" raised a TypeError and was converted to a JSON string '{value}'.")
        # update the attributes of the dataset
        if attrs is not None:
            self.update_attributes(attrs, fullpath)

    def update_group(self, name: str, path: str = "",
                     attrs: Union[Dict, None] = None, **kwargs):
        """
        Update the group 'name' at 'path' (create a new one if 'name' does not exist).
        Optionally, attributes to that group can be given.

        :param name:
        :param value:
        :param path:
        :param attrs:
        :param kwargs: will be passed to create_group()
        :return:
        """

        f = self.file
        fullpath = path + '/' + name
        # create a new group if it does not exist
        if name not in f[path + '/']:
            logging.debug(f"<{self.filename}> Group [{name}] did not exist in {path + '/'}: New group created.")
            f.create_group(path + '/' + name, **kwargs)
            if attrs is not None:
                self.update_attributes(attrs, fullpath)

        # update the attributes of the group
        elif attrs is not None:
            # check to update the attributes of that group
            self.update_attributes(attrs, fullpath)

    def _save_data_from_dict(self, path: str, key: str, d: Any, **kwargs):
        """
        Recursively write the data of a (nested) dict to a hdf5 data set using h5py.

        :param f:
        :param path:
        :param d:
        :return:
        """
        if isinstance(d, dict):
            self.update_group(key, path)
            path = path + "/" + key
            for (d_key, value) in d.items():
                self._save_data_from_dict(str(path), str(d_key), value, **kwargs)
        else:
            self.update_dataset(key=key, value=d, path=path, **kwargs)

    def write_dict(self, d: Dict):
        """
        Write a dictionary to a HDF5 file. The dictionary keys will be used as (nested) groups inside the file.
        :param d:
        :return:
        """
        for (key, value) in d.items():
            self._save_data_from_dict("", str(key), value)

    def read_as_dict(self, d: h5py.Group) -> Dict:
        """
        Recursively read in the hdf5 Group into a dict. And return the final result.

        :param d:
        :return:
        """
        return self._read_in_dict(d)

    def _read_in_dict(self, d: h5py.Group) -> Dict:
        """
        Recursively read in the hdf5 data set into a dict. And return the final result.

        :param d:
        :return:
        """
        if isinstance(d, h5py.Group):
            return_d = {}
            for (key, value) in d.items():
                if isinstance(value, h5py.Group):
                    return_d[key] = self._read_in_dict(value)
                else:
                    return_d[key] = value[()]
        else:
            return_d = d[()]

        return return_d

    def read_dict(self) -> Dict:
        """
        Read an hdf5 file as dictionary
        :return:
        """
        d = {}
        for (key, value) in self.file.items():
            d[key] = self._read_in_dict(value)

        return d


def json_default_converter(o):
    """
    Helper function to serialize some custom object with JSON
    :param o:
    :return:
    """
    if isinstance(o, datetime):
        return o.__str__()
    if isinstance(o, np.ndarray) or isinstance(o, Coordinate):
        return o.tolist()
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.int32):
        return int(o)
    else:
        return o.__dict__
