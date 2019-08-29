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
import json
import numpy as np
import texttable
import logging
import re

from os import listdir, getcwd, mkdir
from os.path import join, isfile, isdir
from datetime import datetime
from typing import Dict, Optional, List

from LayerModel_lib.general import save_file, dict_equal


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
        return "version%d.sim" % self.version

    @property
    def required_keys(self):
        """
        Define the required keys for the results dictionary
        :return:
        """
        return ['name', 'created_on', 'description', 'parameters', 'readme', 'created_by']

    def __bool__(self) -> bool:
        """
        Object evaluates as True only if the model_name is not set to 'empty'

        :return:
        """
        if self.model_name is 'empty':
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

        sim_filename = "version%d.sim" % self.version
        sim_path = join(self.path, sim_filename)
        if isfile(sim_path):
            data = pickle.load(open(sim_path, "rb"))
            # import the loaded data into the same instance of this class
            self.__dict__.clear()
            self.__dict__.update(data)
            # reset the working directory to the value that was set before the data was loaded.
            # As the stored absolute path will most likely be different on different machines
            self.working_directory = new_work_dir

        # load all available results
        res_file_ending = '.result'
        res_file_list = [f for f in listdir(self.path)
                         if isfile(join(self.path, f))
                         and f.endswith(res_file_ending)
                         and f.startswith('version' + str(self.version))]

        # sort the list with .result-files according to the result numbering
        # key searches for the second digit in the file name (which is the result number)
        res_file_list = sorted(res_file_list, key = lambda x: int(re.findall(r'\d+', x)[1]))

        for file in res_file_list:
            res_path = join(self.path, file)
            data = pickle.load(open(res_path, "rb"))
            self.results.append(data)

        # reset the working directory to the value that was set before the data was loaded.
        # As the stored absoulte path will most likely be different on different machines
        self.working_directory = new_work_dir

    def __init__(self, task: str,
                 model_name: str='empty',
                 model_type: str='trunk',
                 scenario: str='empty',
                 version: int=0):
        """
        Loads an existing simulation scenario or creates an empty one if the specified does not exist.

        :param str task: determines if a new scenarios is created (task='create')
                     or it is tried to load (task='load') an existing one.
        :param str model_name: Name of the voxel model that is used in this scenario
        :param str model_type:  the type of the VoxelModel to use, e.g. 'trunk' or 'complete'.
        :param str scenario: Name of the scenario
        :param int version: Version of the scenario

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

            logging.info('New scenario with version %d created..' % self.version)

        elif task == 'load':
            # Try to load an existing scenario

            # check if a file for this scenario exists
            path = join(self.path, self.filename)
            if isfile(path):
                # load scenario if the file exists
                self._load_scenario()
                logging.info('Scenario loaded..')
            else:
                raise FileNotFoundError('Scenario %s not found in %s.' % (self.filename, self.path))
        else:
            raise ValueError('task has to be either "load" or "create"')

        # In any case print the scenario info of the created/loaded scenario
        self.print_scenario_info()

    def save(self):
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

        :return:
        """

        logging.info('Saving the scenario..')

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
            res_filename = "version%d__res%d__%s.result" % (self.version, i, result['name'])
            res_path = join(self.path, res_filename)
            if isfile(res_path):
                stored_data = pickle.load(open(res_path, "rb"))
                # check if the existing file is equal to the result that is present in the SimulationScenario
                if dict_equal(stored_data, result):
                    logging.info("%s did not change. Nothing done." % res_filename)
                else:
                    save_file(result, res_filename, self.path)
            else:
                save_file(result, res_filename, self.path)

        # empty the list of all results
        self.results = []

        # now store the SimulationScenario itself
        sim_filename = self.filename
        sim_path = join(self.path, sim_filename)
        if isfile(sim_path):
            stored_data = pickle.load(open(sim_path, "rb"))
            # check if the existing file is equal to the result that is present in the SimulationScenario
            if dict_equal(stored_data, self.__dict__):
                logging.info("%s did not change. Nothing done." % sim_filename)
            else:
                save_file(self.__dict__, sim_filename, self.path)
        else:
            save_file(self.__dict__, sim_filename, self.path)

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
                        ["#Results", len(self.results)],
                        ["Path", join(self.path, self.filename)]], header=False)

        return table.draw()

    def print_scenario_info(self):
        print(self.get_scenario_info())

    @staticmethod
    def create_results_data_dict(name: str,
                                 created_on: datetime,
                                 created_by: str,
                                 description: str,
                                 parameters: Dict,
                                 readme: str) -> Dict:
        """
        Create a dictionary that is used to store the results.

        :param name: A short name of the results (will be used as filename)
        :param datetime created_on:   Time stamp (datetime.today()) of the results
        :param str created_by:   function/script which generated the results
        :param str description:  a short description of the results
        :param dict parameters :   all the relevant parameters for the results (e.g. frequency, noise power etc)
        :param str readme:       Description on how to read the results, e.g. what is the key in this dictionary
                                that holds the actual results. What is the dimension of the resulting matrix or similar.
        :return:
        """

        return {'name': name,
                'created_on': created_on,
                'created_by': created_by,
                'description': description,
                'parameters': parameters,
                'readme': readme}

    def add_result(self, results_data: Dict):
        """
        Adds the results presented in results_data to self.results.

        :param results_data: A dictionary that should contain several keys. For detailed explanation look at the doc
                            string of create_results_data_dict()
        :return:
        """

        for key in self.required_keys:
            if key not in results_data or results_data[key] == "":
                if key == 'created_on':
                    raise KeyError("results_data needs to have a non-empty '%s' datetime entry!" % key)
                else:
                    raise KeyError("results_data needs to have a non-empty '%s' string entry!" % key)

        # if all keys are set append the data
        self.results.append(results_data)

        # save the new result
        # Check if the directory for the scenario exists
        if not isdir(self.path):
            mkdir(self.path)
            logging.info("Folder for scenario did not exist.\n Created %s" % self.path)

        res_filename = "version%d__res%d__%s.result" % (self.version, len(self.results)-1, results_data['name'])
        save_file(results_data, res_filename, self.path)

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
                params += "%s : %.2e \n" % (key, value)

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
                if key not in self.required_keys:
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

    def find_result(self, **kwargs) -> Dict:
        """
        Return the entry of the result dictionary, depending on the search terms which are given in **kwargs.
        All **kwargs will be AND concatenated.

        :param kwargs:
        :return:
        """
        index = self.find_result_index(**kwargs)

        return self.results[index]

    def find_result_index(self, **kwargs) -> int:
        """
        Return the index of the result dictionary, depending on the search terms which are given in **kwargs.
        All **kwargs will be AND concatenated.
        :param kwargs:
        :return:
        """
        r_list = []

        for (i, r) in enumerate(self.results):
            for key, value in kwargs.items():
                if key.endswith("_not"):
                    # negate the key
                    key = key[0:-4]

                    if key in r and value in r[key]:
                        break
                    elif key in r['parameters'] and r['parameters'][key] == value:
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

        if len(r_list) > 1:
            logging.warning("More than one entry found in results dictionary. Returning the first entry of the list.")

        if len(r_list) > 0:
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
