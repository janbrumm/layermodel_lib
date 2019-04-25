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
import os

from datetime import datetime, timedelta
from itertools import product
from multiprocessing import Pool
from typing import Dict, Tuple, Optional, List

from LayerModel_lib import VoxelModel, TissueProperties, LayerModel, SimulationScenario, Coordinate


class VoxelModelSimulator:
    """
    This class provides functions for parallel calculation of various quantities, e.g. the path loss of a specific
    voxel model.
    """

    def __init__(self, scenario: SimulationScenario,
                 n_samples: int,
                 f_start: float, f_end: float,
                 noise_power_density: Optional[float] = None,
                 transmit_power: Optional[float] = None,
                 test_mode: bool = False,
                 params: Optional[Dict] = None,
                 **kwargs):
        """
        Initialize the simulation.
        :param scenario:    Simulation Scenario to use
        :param f_start:             lower frequency for calculation
        :param f_end:               upper frequency for calculation
        :param noise_power_density: The noise power density used to calculate the channel capacity
        :param transmit_power:      The transmit power used to calculate the channel capacity
        :param n_samples:           number of samples to use for the calculation
        :param params:              Additional parameters that should be written into the results file
        """

        self.params = params
        self.test_mode = test_mode
        self.scenario = scenario
        self.voxel_model = VoxelModel(scenario.model_name)
        self.f_start = f_start
        self.f_end = f_end
        self.n_samples = n_samples
        self.tissue_properties = TissueProperties()
        self.path_loss = None  # set in self.calculate_path_loss_matrix
        self.capacity = None  # used in VoxelModelSimulator.calculate_path_loss_and_capacity_matrix()
        self.distance = None  # set in self.calculate_path_loss_matrix
        # to calculate the channel capacity we also need the noiser power density and the transmit power:
        self.noise_density = noise_power_density
        self.transmit_power = transmit_power
        # all remaining arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def save_path_loss_results(self):
        """
        Save the results of the path loss computation in the simulation scenario file
        """
        # create a dictionary containing the results
        results = self.scenario.create_results_data_dict(
            created_on=datetime.today(),
            created_by=__file__,
            name='path_loss_lm',
            description="Path Loss in the range %.1f-%.1f GHz"
                        % (self.f_start/1e9, self.f_end/1e9),
            parameters={'f_start': self.f_start,
                        'f_end': self.f_end,
                        'n_samples': self.n_samples},
            readme="The key 'path_loss' contains the path loss "
                   "results between each endpoint i and startpoint "
                   "k in entry path_loss[i, k]. The key 'distance'"
                   "contains the corresponding distance between TX"
                   "and RX for each pair of i and k.")

        # add the results to the dictionary
        results['path_loss'] = self.path_loss
        results['distance'] = self.distance

        # add the results to the scenario
        self.scenario.add_result(results_data=results)

    def get_path_loss(self, startpoint: Coordinate, endpoint: Coordinate) -> Dict[str, float]:
        """
        Wrapper function around LayerModel.path_loss for the usage with pool.starmap.

        :param Coordinate startpoint:  Startpoint coordinate
        :param Coordinate endpoint:    Endpoint coordinate

        :return:
            A dictionary containing the path_loss and the distance between the two given coordinates
        """
        lm = LayerModel(self.voxel_model, startpoint, endpoint, self.tissue_properties)

        path_loss = lm.path_loss(self.f_start, self.f_end, self.n_samples)
        distance = lm.distance

        return {'path_loss': path_loss, 'distance': distance}  # , 'endpoint': endpoint, 'startpoint': startpoint}

    def calculate_path_loss_matrix(self, return_values: bool = False, num_workers: int = os.cpu_count())\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the path loss in the given frequency band for all combinations of startpoints and endpoints.
        Implementation is running in parallel on as many available CPUs as possible.

        :param bool return_values:   If set to True, path_loss and distance are returned and no file is saved.
                                Default is False, the values are saved to an .npz file, but not returned
        :param int num_workers:     The number of workers for the parallel pool. Default= os.cpu_count()

        :return:
            numpy.ndarray path_loss     array with shape (len(startpoints), len(endpoints))
                                        containing the path loss between
                                        startpoint(i) and endpoint(j) at path_loss[i, j]
            numpy.ndarray distance      array with shape (len(endpoints), len(startpoints))
                                        the corresponding distance of the link
        """

        self.path_loss = np.zeros((len(self.scenario.startpoints), len(self.scenario.endpoints)))
        self.distance = np.zeros((len(self.scenario.startpoints), len(self.scenario.endpoints)))

        chunksize = 200
        with Pool(processes=num_workers) as pool:
            for ind, res in enumerate(pool.starmap(self.get_path_loss,
                                                   product(self.scenario.startpoints, self.scenario.endpoints),
                                                   chunksize=chunksize)):
                self.path_loss.flat[ind] = res['path_loss']
                self.distance.flat[ind] = res['distance']

        if return_values:
            return self.path_loss, self.distance
        else:
            self.save_path_loss_results()

    def save_path_loss_and_capacity_results(self):
        """
        Save the results of the path loss and capacity computation in the simulation scenario file
        """
        # if radiation loss is included change the filename
        filename = 'path_loss_and_capacity'
        if hasattr(self, 'additional_losses') and hasattr(self, 'receiver_area'):
            filename = filename + '_radiation_loss_' + str(int(np.round(self.receiver_area * 10000))) + 'cm^2'
        if hasattr(self, 'filename_suffix'):
            filename = filename + "_" + self.filename_suffix

        # create a dictionary containing the results
        results = self.scenario.create_results_data_dict(
            created_on=datetime.today(),
            created_by=__file__ + "=> calculate_path_loss_and_capacity_matrix()",
            name=filename,
            description="Path Loss and Capacity in the range %.1f-%.1f GHz"
                        % (self.f_start/1e9, self.f_end/1e9),
            parameters={'f_start': self.f_start,
                        'f_end': self.f_end,
                        'n_samples': self.n_samples,
                        'noise_density': self.noise_density,
                        'transmit_power': self.transmit_power},
            readme="The key 'path_loss' contains the path loss "
                   "results between each endpoint i and startpoint "
                   "k in entry path_loss[k, i]. The key 'capacity' contains"
                   "the channel capacity between endpoint i and startpoint k."
                   "The key 'distance'"
                   "contains the corresponding distance between TX"
                   "and RX for each pair of i and k.")

        # add additional parameters from self.params
        for key, value in self.params.items():
            results['parameters'][key] = value

        # add the results to the dictionary
        results['path_loss'] = self.path_loss
        results['distance'] = self.distance
        results['capacity'] = self.capacity

        # add the results to the scenario
        self.scenario.add_result(results_data=results)

    def get_path_loss_and_capacity(self, startpoint: Coordinate, endpoint: Coordinate) -> Dict:
        """
        Wrapper function around LayerModel.path_loss and LayerModel.channel_capacity() for the usage with pool.starmap.

        :param Coordinate startpoint:  Startpoint coordinate
        :param Coordinate endpoint:    Endpoint coordinate

        :return:
            A dictionary containing the path_loss and the distance between the two given coordinates
        """
        lm = LayerModel(self.voxel_model, startpoint, endpoint, self.tissue_properties,
                        model_type=self.scenario.model_type)

        # calculate S21 and use it for both path loss and capacity calculation.
        (s21, f) = lm.S21(self.f_start, self.f_end, self.n_samples)

        # if additional_losses is set to radiation loss include the radiation loss
        if hasattr(self, "additional_losses") and hasattr(self, "receiver_area"):
            if self.additional_losses == "radiation loss":
                receiver_area = self.receiver_area
                logging.debug("Applied radiation loss with effective area= %f" % receiver_area)
                distance = lm.distance
                rl = receiver_area / (4 * np.pi * distance ** 2)
                s21 = np.sqrt(rl) * s21
        if hasattr(self, "additional_losses") and hasattr(self, "loss_function"):
            if self.additional_losses == "free space loss":
                logging.debug("Applied isotropic free space loss")
                distance = lm.distance
                fsl = self.loss_function(distance, f)
                s21 = np.sqrt(fsl) * s21

        # calculate the path loss and the capacity using the S21 calculated above
        path_loss = lm.path_loss(self.f_start, self.f_end, self.n_samples, precalculated_S21_f=(s21, f))
        capacity = lm.channel_capacity(self.noise_density, self.transmit_power,
                                       f_start=self.f_start, f_end=self.f_end,
                                       n_samples=self.n_samples, precalculated_S21_f=(s21, f))
        distance = lm.distance

        return {'path_loss': path_loss, 'capacity': capacity, 'distance': distance}

    def calculate_path_loss_and_capacity_matrix(self, num_workers: int = os.cpu_count()):
        """
        Calculate the path loss and the channel capacity in the given frequency band for all combinations of
        startpoints and endpoints.
        Implementation is running in parallel on as many available CPUs as possible.

        :param int num_workers:     The number of workers for the parallel pool. Default= os.cpu_count()

        :return:
            numpy.ndarray path_loss     array with shape (len(startpoints), len(endpoints))
                                        containing the path loss between
                                        startpoint(i) and endpoint(j) at path_loss[i, j]
            numpy.ndarray distance      array with shape (len(endpoints), len(startpoints))
                                        the corresponding distance of the link
        """

        if self.test_mode:
            # for debug select the first 10 start and endpoints
            startpoints = self.scenario.startpoints[0:10]
            endpoints = self.scenario.endpoints[0:10]
        else:
            startpoints = self.scenario.startpoints
            endpoints = self.scenario.endpoints

        self.path_loss = np.zeros((len(startpoints), len(endpoints)))
        self.capacity = np.zeros((len(startpoints), len(endpoints)))
        self.distance = np.zeros((len(startpoints), len(endpoints)))

        start_time = datetime.today()
        logging.info("%s -- Started calculation of Path Loss and Capacity.." % str(start_time))

        chunksize = 200
        with Pool(processes=num_workers) as pool:
            for ind, res in enumerate(pool.starmap(self.get_path_loss_and_capacity,
                                                   product(startpoints, endpoints),
                                                   chunksize=chunksize)):
                self.path_loss.flat[ind] = res['path_loss']
                self.distance.flat[ind] = res['distance']
                self.capacity.flat[ind] = res['capacity']

        end_time = datetime.today()
        duration = end_time - start_time
        logging.info("%s -- Finished calculation after %s.." % (str(end_time), str(duration)))
        self.save_path_loss_and_capacity_results()

    def calculate_power_delay_profile(self, threshold_dB: float = 30,
                                      equivalent_baseband: Dict = None,
                                      endpoint_indices: Optional[List] = None) \
            -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Calculate the power delay profile and the RMS delay spread for the given scenario

        :param float threshold_dB: The threshold up to which decay of the power delay profile the values shall be
                            included in the calculation of the rms delay spread and the power delay profile.
        :param equivalent_baseband: Parameter for LayerModel.impulse_response for conversion to equivalent baseband
        :param list endpoint_indices: Optional parameter to calculate the PDP only for specific indices listed

        :rtype: Tuple[np.ndarray, np.ndarray, float, float]
        :return: a tuple with the following elements: the power_delay_profile, the delay vector,
                                                      the rms delay spread and the average delay.
        """

        # sampling rate cannot be chosen arbitrarily when converting to equivalent baseband
        if equivalent_baseband is not None:
            f_sample = equivalent_baseband['B']
        else:
            f_sample = 20e9

        # start with an empty PDP
        power_delay_profile = np.zeros((0,))

        startpoints = self.scenario.startpoints

        if endpoint_indices is not None:
            # use only the elements from the endpoints that are specified in endpoint_indices
            endpoints = [self.scenario.endpoints[i] for i in endpoint_indices]
        else:
            endpoints = self.scenario.endpoints

        # Debug mode
        if self.test_mode:
            startpoints = startpoints[0:10]
            endpoints = endpoints[0:10]
            log_interval = timedelta(seconds=2)
        else:
            log_interval = timedelta(hours=2)

        start_time = datetime.today()
        last_time = start_time
        logging.info("%s -- Started calculation of Power Delay Profile.." % str(start_time))

        total_length = (len(startpoints) * len(endpoints))

        for index, (sp, ep) in enumerate(product(startpoints, endpoints)):
            lm = LayerModel(self.voxel_model, sp, ep, model_type=self.scenario.model_type)
            if equivalent_baseband is None:
                f_start = self.f_start
                f_end = self.f_end
            else:
                # define variables for downconversion
                bandwidth = equivalent_baseband['B']
                carrier_frequency = equivalent_baseband['fc']
                f_start = carrier_frequency - bandwidth / 2
                f_end = carrier_frequency + bandwidth / 2
                f_sample = bandwidth

            # calculate S21 and use it for both path loss and capacity calculation.
            (s21, f) = lm.S21(f_start, f_end, self.n_samples)

            # if additional_losses is set to free space loss include the free space loss
            if hasattr(self, "additional_losses") and hasattr(self, "loss_function"):
                if self.additional_losses == "free space loss":
                    logging.debug("Applied isotropic free space loss")
                    distance = lm.distance
                    fsl = self.loss_function(distance, f)
                    s21 = np.sqrt(fsl) * s21

            impulse_response = lm.impulse_response(normalize_delay='edge',
                                                   n_samples=self.n_samples,
                                                   threshold_dB=threshold_dB,
                                                   f_sample=f_sample,
                                                   equivalent_baseband=equivalent_baseband,
                                                   precalculated_S21_f=(s21, f))[0]

            if index == 0:
                power_delay_profile = abs(impulse_response)**2
            else:
                # if the current impulse_response is longer than the power_delay_profile append as many zeros as needed
                diff_size = impulse_response.size - power_delay_profile.size
                if diff_size > 0:
                    power_delay_profile = np.pad(power_delay_profile, [(0, diff_size), (0, 0)],
                                                 mode='constant', constant_values=0)
                elif diff_size < 0:
                    diff_size = -diff_size
                    impulse_response = np.pad(impulse_response, [(0, diff_size), (0, 0)],
                                              mode='constant', constant_values=0)

                power_delay_profile += abs(impulse_response)**2

            now = datetime.today()
            if now - last_time > log_interval:
                logging.info("%s -- PDP Calculation %d/%d = %.1f%% .." % (str(now), index, total_length,
                                                                          index/total_length*100))
                last_time = now

        # calculate the average
        power_delay_profile = power_delay_profile / total_length

        end_time = datetime.today()
        duration = end_time - start_time
        logging.info("%s -- Finished calculation of Power Delay Profile after %s.." % (str(end_time), str(duration)))
        logging.info("%s -- Starting RMS Delay Spread Calculation.." % str(datetime.today()))

        power_delay_profile.shape = (-1,)
        """ cut the power delay profile after it has reached a decay of more than threshold_dB dB """
        # find the maximum
        pdp_max_ind = np.argmax(power_delay_profile)
        pdp_max = power_delay_profile[pdp_max_ind]
        # determine the threshold value
        threshold_value = pdp_max * 10**(-threshold_dB/10)
        # find the index where the threshold is first met after the PDP maximum
        threshold_index = np.argmax(power_delay_profile[pdp_max_ind::] < threshold_value)
        # the threshold_index is relative to the maximum of the pdp -> add pdp_max_ind
        power_delay_profile = power_delay_profile[0:(threshold_index + pdp_max_ind)]

        # generate delay vector
        tau = np.arange(0, power_delay_profile.size) / f_sample
        tau.shape = (-1,)

        # calculate the RMS delay spread
        tau_mean = np.trapz(power_delay_profile * tau, tau) / np.trapz(power_delay_profile, tau)

        tau_rms = np.sqrt(np.trapz((tau - tau_mean)**2 * power_delay_profile, tau) /
                          np.trapz(power_delay_profile, tau))

        # adjust the values such that the first entry of the PDP is 1 == 0dB
        power_delay_profile /= pdp_max

        return power_delay_profile, tau, tau_rms, tau_mean
