# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
import pickle
import h5py
import json
import numpy as np
import texttable
import logging
import re

from os import listdir, getcwd, mkdir, remove
from os.path import join, isfile, isdir
from datetime import datetime
from typing import Dict, Optional, List, Union

from LayerModel_lib.hdf5 import HDF5
from LayerModel_lib.general import save_file, dict_equal
from LayerModel_lib.coordinate import Coordinate


class SimulationScenario:
    """
    This class describes a simulation scenario for a specific voxel model with scenario
    specific start and endpoints of the communication.

    All scenario files are stored in the working_directory, which has to be set before the first usage)
     with the following scheme:
    [model_name]__[scenario]__[version].sim
    where the version is only added if a file with the same name already exist.
    """

    # Initially set an empty working directory
    working_directory = ""

    @property
    def path(self):
        """
        Construct the full path to save the scenario to or load it from
        :return:
        """
        if self.working_directory == "":
            self.working_directory = getcwd()
            logging.warning("No working directory set. It was set to \n %s" % self.working_directory)

        return join(self.working_directory, self.model_name + "__" + self.scenario)

    @property
    def filename(self):
        """
        Define the filename of the scenario
        :return:
        """
        file_suffix = '.h5'
        return f"version{self.version}{file_suffix}"

    @property
    def required_result_keys(self):
        """
        Define the required keys for each of the results dictionary
        :return:
        """
        return ['name', 'created_on', 'description', 'parameters', 'readme', 'created_by', 'parameters',
                'store_dict_as_nested_dataset']  # a list of all results that are dicts and to be save as a nested
        # datasets. If a dict is not in this list it is converted into a JSON string and the stored.

    @property
    def required_scenario_attributes(self):
        """
        Define the required attributes for the SimulationScenario itself. This
        is used to determine which part of the SimulationScenario is saved as attribute in the HDF5 file.
        :return:
        """
        return ['model_name', 'created_on', 'model_type', 'scenario', 'scenario_description', 'version', 'parameters']

    def __bool__(self) -> bool:
        """
        Object evaluates as True only if the model_name is not set to 'empty'

        :return:
        """
        if self.model_name == 'empty':
            return False
        else:
            return True

    def _load_scenario(self):
        """
        Load the desired scenario
        :return:
        """
        # store the currently set working_directory
        new_work_dir = self.working_directory

        # check for files that are in the simulation scenario folder but not yet part of the scenario
        # or files that have been deleted but are still stored in the scenario.
        self.check_result_files()

        sim_filename = f"version{self.version}.h5"
        sim_path = join(self.path, sim_filename)
        # logging.info(f"{sim_path}")
        with h5py.File(sim_path, 'r') as f:
            hdf = HDF5(f)
            # Start and Endpoints
            sp = f['startpoints'][:]
            self.startpoints = [Coordinate(sp[i, :]) for i in range(sp.shape[0])]
            ep = f['endpoints'][:]
            self.endpoints = [Coordinate(ep[i, :]) for i in range(ep.shape[0])]
            # Read in the metadata from the attributes of the results group
            for a_key, a_val in f['/results'].attrs.items():
                # convert byte array stored in HDF5 file to Python 3 str object
                if isinstance(a_val, bytes):
                    a_val = a_val.decode("utf-8")

                if a_key == 'created_on':
                    a_val = datetime.strptime(a_val, '%Y-%m-%d %H:%M:%S.%f')
                # check if the value was stored as JSON string
                elif isinstance(a_val, str) and a_val.startswith('JSON='):
                    a_val = json.loads(a_val[5::])

                if a_key.startswith('param_'):
                    self.parameters[a_key[6::]] = a_val
                else:
                    setattr(self, a_key, a_val)

            # Read in all the results:
            for result in f['/results']:
                r = {}
                for a_key, a_val in f['/results/' + result].attrs.items():
                    # convert byte array stored in HDF5 file to Python 3 str object
                    if isinstance(a_val, bytes):
                        a_val = a_val.decode("utf-8")
                    elif isinstance(a_val, np.int32) and a_val.size == 1:
                        a_val = a_val.tolist()  # converts scalar numpy values to python type
                    elif isinstance(a_val, np.float64):
                        a_val = float(a_val)

                    if a_key == 'created_on':
                        a_val = datetime.strptime(a_val, '%Y-%m-%d %H:%M:%S.%f')

                    # check if the value was stored as JSON string
                    elif isinstance(a_val, str) and a_val.startswith('JSON='):
                        a_val = json.loads(a_val[5::])

                    if a_key.startswith('param_'):
                        if 'parameters' not in r:
                            r['parameters'] = {}
                        r['parameters'][a_key[6::]] = a_val  # remove leading 'param_'
                    else:
                        r[a_key] = a_val
                # read in all the data sets
                for d_key, d_val in f['/results/' + result].items():
                    if 'store_dict_as_nested_dataset' in r and\
                            d_key in list(r['store_dict_as_nested_dataset']):
                        r[d_key] = hdf.read_as_dict(d_val)
                    else:
                        if isinstance(d_val, h5py.Dataset):
                            value = d_val[()]
                            # check if the value was stored as JSON string
                            if isinstance(value, bytes):
                                value = value.decode("utf-8")
                                if value.startswith("JSON="):
                                    value = json.loads(value[5::])

                            # otherwise store it directly
                            r[d_key] = value

                self.results.append(r)

        # reset the working directory to the value that was set before the data was loaded.
        # As the stored absoulte path will most likely be different on different machines
        self.working_directory = new_work_dir

    def __init__(self, task: str,
                 model_name: str = 'empty',
                 model_type: str = 'trunk',
                 scenario: str = 'empty',
                 version: int = 0,
                 silent: bool = False):
        """
        Loads an existing simulation scenario or creates an empty one if the specified does not exist.

        :param str task: determines if a new scenarios is created (task='create')
                     or it is tried to load (task='load') an existing one.
        :param str model_name: Name of the voxel model that is used in this scenario
        :param str model_type:  the type of the VoxelModel to use, e.g. 'trunk' or 'complete'. Does NOT have any
                                functionality.
        :param str scenario: Name of the scenario
        :param int version: Version of the scenario
        :param silent: do not print a message after loading or creating of the scenario.
        """

        if '__' in model_name or '__' in scenario:
            raise ValueError('Neither model_name nor scenario are allowed to contain two underscores\'__\'')

        self.model_name = model_name  # name of the voxel model
        self.scenario = scenario  # short title of the scenario
        self.model_type = model_type  # type of the voxel model, e.g. 'trunk' or 'complete'
        self.scenario_description = ''  # extended description of the scenario setup.
        self.startpoints = []  # position of all the transmit antennas
        self.endpoints = []  # position of  all the receive antennas
        self.results = []  # an empty list for storing results associated with this scenario
        self.version = version  # version of this scenario
        self.created_on = datetime.today()  # date this scenario was created on
        self.parameters = {}  # dict to store additional parameters for the scenario

        self.updated_results = []  # a list of indices of all the results that have been updated and should be saved.
        # Only results are stored as HDF5 file if their index is in the list above!

        if task == 'create':
            # Create a new simulation scenario, if there is already one with the same model_name and scenario string
            # increment the version

            # increment the version as long as no scenario with this version number exists
            exist = True
            while exist:
                sim_file = join(self.path, 'version' + str(self.version) + '.sim')
                if isfile(sim_file):
                    self.version += 1
                else:
                    exist = False

            if not silent:
                logging.info('New scenario with version %d created..' % self.version)

        elif task == 'load':
            # Try to load an existing scenario

            # check if a file for this scenario exists
            path = join(self.path, self.filename)
            if isfile(path):
                # load scenario if the file exists
                self._load_scenario()
                if not silent:
                    logging.info('Scenario loaded..')
            else:
                raise FileNotFoundError('Scenario %s not found in %s.' % (self.filename, self.path))
        else:
            raise ValueError('task has to be either "load" or "create"')

        if not silent:
            # In any case print the scenario info of the created/loaded scenario
            self.print_scenario_info()

    @staticmethod
    def create(model_name: str = 'empty', scenario: str = 'empty', model_type: str = 'empty',
               silent: bool = False) -> 'SimulationScenario':
        """
        Wrapper for self.__init__('create', model_name, scenario):

        :param model_name: Name of the voxel model that is used in this scenario
        :param scenario: Name of the scenario
        :param silent: do not print a message after creating of the scenario.
        :return:
        """
        return SimulationScenario('create', model_name=model_name, model_type=model_type, scenario=scenario,
                                  silent=silent)

    @staticmethod
    def load(model_name: str = 'empty', scenario: str = 'empty', version: int = 0, model_type: str = 'empty',
             silent: bool = False) -> 'SimulationScenario':
        """
        Wrapper for __init__('load', model_name, scenario, version: int=0):
        :param model_name: Name of the voxel model that is used in this scenario
        :param scenario: Name of the scenario
        :param version: Version of that scenario to load
        :param silent: do not print a message after loading of the scenario.
        :return:
        """
        return SimulationScenario('load', model_name=model_name, scenario=scenario, version=version,
                                  model_type=model_type, silent=silent)

    def save(self, force_overwrite: bool = False):
        """
        Saves the current scenario. For that purpose the scenario is split into multiple files to allow easier
        handling with git.
        The file and folder structure is the following:

        SimulationScenario.working_directory /
            self.model_name __ self.scenario /              <-- A folder for storing all results for a scenario
                version0.sim                        <-- The pickled version of SimulationScenario
                                                        (without an empty list of results dictionaries)
                version0_res0_PathLoss.result          <-- pickled version of entry 0 of the results list
                version0_res1_PowerDelayProfile.result <-- pickled version of entry 1 of the results list

        :param force_overwrite: open the HDF5 file in write mode if True (defaults to append). Hence, forcing any
                                value to be overwritten.
        :return:
        """

        logging.info(f'Saving the scenario {self.model_name}/{self.scenario} ..')

        if not self.scenario_description:
            logging.warning('Scenario description is empty !!\n')
        if len(self.startpoints) == 0:
            logging.warning('No startpoints set !!\n')
        if len(self.endpoints) == 0:
            logging.warning('No endpoints set !!\n')

        # Check if the directory for the scenario exists
        if not isdir(self.path):
            mkdir(self.path)
            logging.info("Folder for scenario did not exist.\n Created %s" % self.path)

        # store the basic scenario
        sim_filename = self.filename
        sim_path = join(self.path, sim_filename)

        if force_overwrite:
            io_mode = 'w'
        else:
            io_mode = 'a'

        with h5py.File(sim_path, io_mode) as f:
            hdf = HDF5(f, compressed=True)
            attrs = {'model_name': self.model_name, 'model_type': self.model_type, 'unit': 'mm'}
            hdf.update_dataset('startpoints', self.startpoints, attrs=attrs)
            hdf.update_dataset('endpoints', self.endpoints, attrs=attrs)

            attrs = {}
            for a in self.required_scenario_attributes:
                if a == 'created_on':
                    attrs[a] = str(getattr(self, a, ""))
                elif a == 'parameters':
                    for p_key, p_val in self.parameters.items():
                        attrs["param_" + p_key] = p_val
                else:
                    attrs[a] = getattr(self, a, "")

            hdf.update_group('results', attrs=attrs)

            # store all the results
            for res_idx, result in enumerate(self.results):
                res_filename = f"version{self.version}__{result['name']}.h5"
                if res_idx in self.updated_results:  # only update if the index is in updated_results
                    with h5py.File(join(self.path, res_filename), io_mode) as f_res:
                        # create a group result in the external file
                        hdf_res = HDF5(f_res, compressed=True)

                        # collect all metadata for storing as attributes for the group 'result'
                        attrs = {}
                        for r_key, r_val in result.items():
                            if r_key in self.required_result_keys:  # the metadata are all the required keys
                                if r_key == 'created_on':
                                    attrs[r_key] = np.string_(r_val)
                                elif r_key == 'parameters':
                                    for p_key, p_val in r_val.items():
                                        attrs["param_" + p_key] = p_val
                                else:
                                    attrs[r_key] = r_val

                        hdf_res.update_group('result', attrs=attrs)
                        for r_key, r_val in result.items():
                            if r_key not in self.required_result_keys:
                                attr_option = {}
                                if r_key in attrs['store_dict_as_nested_dataset']:
                                    attr_option = {'attrs': {'python_dict_stored_as_nested_dataset': True}}

                                hdf_res.update_dataset(r_key, r_val, path="/result", **attr_option)
                else:
                    logging.debug(f"<{sim_filename}> [/results/{result['name']}]: Nothing changed -> file not saved.")

                if result['name'] not in f['/results']:
                    # add a link to the versionX.h5 file to the corresponding result
                    f[f"results/{result['name']}"] = h5py.ExternalLink(res_filename, "/result")
                    logging.info(f"<{sim_filename}> [/results/{result['name']}]: Link to {res_filename} created.")
                else:
                    logging.debug(f"<{sim_filename}> [/results/{result['name']}]: Link exists already.")

        # all results in updated_results should be saved now -> empty the list
        self.updated_results = []

        # check for files that are in the simulation scenario folder but not yet part of the scenario
        # or files that have been deleted but are still stored in the scenario.
        self.check_result_files()

        logging.info('.. everything saved to disc')

    def check_result_files(self):
        """
        Check the folder of the SimulationScenario for .h5 files not present in the results list and also
        for entries in the results list whose files have been deleted.

        :return:
        """
        sim_filename = f"version{self.version}.h5"
        sim_path = join(self.path, sim_filename)

        # all files containing results in the folder of the scenario
        file_list = [f for f in listdir(self.path)
                     if isfile(join(self.path, f)) and f.endswith('.h5') and f.startswith(f'version{self.version}')
                     and f != sim_filename]

        # iterate through all result links to see which files are still present
        with h5py.File(sim_path, 'a') as f:
            for result in f['/results']:
                link_target = f['/results'].get(result, getlink=True).filename

                # test all the results. if one throughs an exception it seems to be deleted
                try:
                    f['/results/' + result]
                except KeyError:
                    logging.warning(f"<{self.model_name}__{self.scenario}/{sim_filename}> [/results/{result}]: "
                                    f"Was not able to open external link to "
                                    f"\"{link_target}\". "
                                    f"Deleted entry in results list")
                    del f['/results/' + result]  # delete the link from the scenario
                else:
                    # remove all existing results from the file list:
                    file_list.remove(link_target)

            # file list contains now only files that are not part of the simulation scenario yet. Hence, add them
            for filename in file_list:
                with h5py.File(join(self.path, filename), 'r') as f_res:
                    result_name = f_res['result'].attrs['name'].decode("utf-8")
                    if result_name not in f['/results']:
                        # add a link to the versionX.h5 file to the corresponding result
                        f[f"results/{result_name}"] = h5py.ExternalLink(filename, "/result")
                        logging.info(f"<{self.model_name}__{self.scenario}/{sim_filename}> "
                                     f"[/results/{result_name}]: New file {filename} found."
                                     f"Created external link.")

    def save_json(self):
        """
        Save Scenario as JSON file
        :return:
        """

        def jdefault(o):
            if isinstance(o, datetime):
                return o.__str__()
            else:
                return o.__dict__

        logging.info('Saving the scenario as JSON..')

        if not self.scenario_description:
            logging.warning('Scenario description is empty !!\n')
        if len(self.startpoints) == 0:
            logging.warning('No startpoints set !!\n')
        if len(self.endpoints) == 0:
            logging.warning('No endpoints set !!\n')

        # Check if the directory for the scenario exists
        if not isdir(self.path):
            mkdir(self.path)
            logging.info("Folder for scenario did not exist.\n Created %s" % self.path)

        # Start saving the results dictionaries
        for (i, result) in enumerate(self.results):
            res_filename = "version%d__res%d__%s.result.JSON" % (self.version, i, result['name'])
            res_path = join(self.path, res_filename)
            with open(res_path, "w") as f:
                json.dump(result, f, default=jdefault)

        # empty the list of all results
        self.results = []

        # now store the SimulationScenario itself
        sim_filename = self.filename + ".JSON"
        sim_path = join(self.path, sim_filename)
        with open(sim_path, "w") as f:
            json.dump(self.__dict__, f, default=jdefault)

    def get_scenario_info(self) -> str:
        """
        Create a string containing all basic information about this scenario
        :return:
        """
        table = texttable.Texttable()

        # this table will not have a header on top but on the left side
        table.set_deco(table.BORDER | table.HLINES | table.VLINES)
        table.set_cols_width([15, 50])

        table.add_rows([["Voxel Model", self.model_name],
                        ["Type of Model", self.model_type],
                        ["Scenario", self.scenario],
                        ["Created on", self.created_on.strftime("%Y-%m-%d")],
                        ["Version", self.version],
                        ["TX Locations", len(self.startpoints)],
                        ["RX Locations", len(self.endpoints)],
                        ["Description", self.scenario_description],
                        ["#Results", len(self.results)]], header=False)

        if hasattr(self, 'parameters') and self.parameters:
            table.add_row(["Parameters", self._generate_param_string(self.parameters)])

        table.add_row(["Path", join(self.path, self.filename)])

        return table.draw()

    def print_scenario_info(self):
        print(self.get_scenario_info())

    @staticmethod
    def create_results_data_dict(name: str,
                                 created_on: datetime,
                                 created_by: str,
                                 description: str,
                                 parameters: Dict,
                                 readme: str,
                                 store_dict_as_nested_dataset: List = None,
                                 **kwargs) -> Dict:
        """
        Create a dictionary that is used to store the results.

        :param name: A short name of the results (will be used as filename)
        :param datetime created_on:   Time stamp (datetime.today()) of the results
        :param str created_by:   function/script which generated the results
        :param str description:  a short description of the results
        :param dict parameters :   all the relevant parameters for the results (e.g. frequency, noise power etc)
        :param str readme:       Description on how to read the results, e.g. what is the key in this dictionary
                                that holds the actual results. What is the dimension of the resulting matrix or similar.
        :param List store_dict_as_nested_dataset: A list of all results that are dicts and to be save as a nested
                                                  datasets. If a dict is not in this list it is converted into a
                                                  JSON string and then stored.
        :return:
        """
        if store_dict_as_nested_dataset is None:
            store_dict_as_nested_dataset = []

        data = {'name': name,
                'created_on': created_on,
                'created_by': created_by,
                'description': description,
                'parameters': parameters,
                'readme': readme,
                'store_dict_as_nested_dataset': store_dict_as_nested_dataset}
        # add all kwargs to the dict as well
        for (key, value) in kwargs.items():
            data[key] = value

        return data

    def add_result(self, results_data: Dict):
        """
        Adds the results presented in results_data to self.results.

        :param results_data: A dictionary that should contain several keys. For detailed explanation look at the doc
                            string of create_results_data_dict()
        :return:
        """

        for key in self.required_result_keys:
            if key not in results_data or results_data[key] == "":
                if key == 'created_on':
                    raise KeyError("results_data needs to have a non-empty '%s' datetime entry!" % key)
                else:
                    raise KeyError("results_data needs to have a non-empty '%s' string entry!" % key)

        # check if name of new results is unique in list of all existing results
        # if it is not unique, append the current date to the name.
        result_names = [r['name'] for r in self.results]
        if results_data['name'] in result_names:
            name_old = results_data['name']
            results_data['name'] = results_data['name'] + f"-{datetime.now():%Y-%m-%d}"
            logging.warning(f"A result with the name '{name_old}'' exists already. "
                            f"Renamed the result to '{results_data['name']}'")

        # if all keys are set append the data
        self.results.append(results_data)
        self.updated_results.append(len(self.results) - 1)  # add the last entry in results to the updated list.

        # save the new result
        # Check if the directory for the scenario exists
        if not isdir(self.path):
            mkdir(self.path)
            logging.info("Folder for scenario did not exist.\n Created %s" % self.path)

        # save the scenario.
        self.save()

    def update_result(self, index: int = None, **kwargs):
        """
        Update a certain result. Makes sure the result index is added to the updated_results list and is hence, saved
        once the save() function is called.
        :param index: Index of the result. If none is given, **kwargs are used to find the entry in results using
                      index = self.find_result_index(**kwargs)
        :param kwargs:
        :return:
        """
        if index is None:
            index = self.find_result_index(**kwargs, raise_on_multiple=True)

        self.updated_results.append(index)
        return self.results[index]

    def delete_result(self, **kwargs):
        """
        Delete the result, given by search parameters as described for self.find_result_index()

        :param kwargs:
        :return:
        """
        # get the index of the result that is to be deleted.
        index = self.find_result_index(raise_on_multiple=True, **kwargs)
        name = self.results[index]['name']

        # delete the link inside the simulation scenario file
        sim_filename = f"version{self.version}.h5"
        sim_path = join(self.path, sim_filename)
        with h5py.File(sim_path, 'a') as f:
            del f['/results/' + name]

        # delete the results file
        res_filename = f"version{self.version}__{name}.h5"
        res_path = join(self.path, res_filename)
        remove(res_path)

        # remove the entry from the results list
        del self.results[index]

    def _generate_param_string(self, parameters: Dict) -> str:
        """
        Generate a string of all parameters for display
        """
        params = ""
        # make a list of all parameters
        for key, value in parameters.items():
            # if value is not a string, assume it is a number.
            if type(value) is str:
                params += "%s : %s \n" % (key, value)
            elif type(value) is int:
                params += "%s : %d \n" % (key, value)
            elif type(value) is list or type(value) is dict:
                params += "%s : %s \n" % (key, str(value))
            else:
                try:
                    params += "%s : %.2e \n" % (key, value)
                except(TypeError, ValueError):
                    try:
                        params += "%s : %s \n" % (key, str(value))
                    except(TypeError, ValueError):
                        logging.warning('Parameter %s could not be printed.' % key)

        return params

    def get_results_overview(self) -> str:
        """
        Show an overview over all stored results.
        :return:
        """
        table = texttable.Texttable()

        # header of the table
        header = ['Index', 'Created on', 'Description', 'Parameters']
        table.header(header)

        for index, r in enumerate(self.results):
            params = self._generate_param_string(r['parameters'])
            table.add_row(['%d' % index,  r['created_on'].strftime("%Y-%m-%d"),
                           r['description'], params])

        table.set_cols_width([5, 10, 20, 40])

        return table.draw()

    def print_result_overview(self):
        print(self.get_results_overview())

    def get_result(self, index: int) -> str:
        """
        Return a string representing all the data that is present for one result[index]
        :param index:
        """

        if index > len(self.results):
            raise IndexError("Index must not be larger than the number of results. "
                             "There are only %d" % len(self.results) + " results stored.")
        else:
            r = self.results[index]
            params = self._generate_param_string(r['parameters'])

            table = texttable.Texttable()
            # this table will not have a header on top but on the left side
            table.set_deco(table.BORDER | table.HLINES | table.VLINES)
            table.set_cols_width([15, 50])
            table.set_cols_dtype(['t', 't'])

            table.add_rows([["Index", index],
                            ["Name", r['name']],
                            ["Created on", r['created_on'].strftime("%Y-%m-%d")],
                            ["Created by", r['created_by']],
                            ["Description", r['description']],
                            ["Parameters", params],
                            ["Readme", r['readme']]], header=False)

            for key, value in r.items():
                if key not in self.required_result_keys:
                    if type(value) is np.ndarray:
                        table.add_row([key, "Numpy.ndarray with shape=%s" % str(value.shape)])
                    elif isinstance(value, str):
                        table.add_row([key, value])
                    elif isinstance(value, int) or isinstance(value, float):
                        s = ("%e" % value)
                        table.add_row([key, s])
                    else:
                        table.add_row([key, str(value)])

            return table.draw()

    def print_result(self, index: int):
        print(self.get_result(index))

    def find_result(self, **kwargs) -> Union[Dict, List[Dict]]:
        """
        Return the entry of the result dictionary, depending on the search terms which are given in **kwargs.
        All **kwargs will be AND concatenated.

        :param kwargs:
        :return:
        """
        index = self.find_result_index(**kwargs)

        if isinstance(index, int):
            return self.results[index]
        else:
            return [self.results[i] for i in index]

    def find_result_index(self, raise_on_multiple: bool = False, find_multiple: bool = False,
                          **kwargs) -> Union[int, List]:
        """
        Return the index of the result dictionary, depending on the search terms which are given in **kwargs.
        All **kwargs will be AND concatenated.
        :param raise_on_multiple: Raise an exception if multiple results have been found
        :param find_multiple: Return a list of all found results if True
        :param kwargs:
        :return:
        """
        r_list = []

        for (i, r) in enumerate(self.results):
            for key, value in kwargs.items():
                if key.endswith("_not"):
                    # negate the key
                    key = key[0:-4]

                    if key not in r and key not in r['parameters']:
                        pass
                    else:
                        break
                else:
                    if key in r and value in r[key]:
                        # found the value in one of the keys
                        pass
                    elif key in r['parameters'] and r['parameters'][key] == value:
                        # found it as one of the parameters
                        pass
                    else:
                        break
            else:
                r_list.append(i)

        if len(r_list) > 1 and not find_multiple:
            text = f"More than one entry found for {kwargs} in results dictionary (indices: {r_list}). "\
                f"Returning the first entry of the list."
            if raise_on_multiple:
                raise ValueError(text)
            else:
                logging.warning(text)

        if len(r_list) > 0:
            if find_multiple:
                return r_list
            else:
                return r_list[0]
        else:
            raise ValueError("No result found for %s." % str(kwargs))

    @staticmethod
    def list_all_scenarios(model_name: Optional[str]=None,
                           scenario: Optional[str]=None) -> List:
        """
        Return a list of all SimulationScenario in the working_directory. Searches for the specified model_name
        and scenario string only.

        :param model_name:
        :param scenario:
        :return:
        """
        if SimulationScenario.working_directory == "":
            if SimulationScenario.working_directory == "":
                SimulationScenario.working_directory = getcwd()
                logging.warning("No working directory set. It was set to \n %s" % SimulationScenario.working_directory)

        file_ending = '.sim'
        file_list = [f for f in listdir(SimulationScenario.working_directory)
                     if isfile(join(SimulationScenario.working_directory, f))
                     and f.endswith(file_ending)]

        info_list = []
        result_list = []
        for f in file_list:
            f_info = f[0:-4].split('__')
            d = {'info_string': f_info[0] + ": " + f_info[1].replace('_', ' ') + ", v. " + f_info[2],
                 'filename': f,
                 'model_name': f_info[0],
                 'scenario': f_info[1],
                 'version': f_info[2]}

            if model_name is not None and model_name != d['model_name']:
                continue
            if scenario is not None and scenario != d['scenario']:
                continue

            result_list.append(d)
            info_list.append(d['info_string'])

        return result_list
