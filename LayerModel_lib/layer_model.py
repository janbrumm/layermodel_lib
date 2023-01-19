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
import matplotlib.pyplot as plt
import logging

from scipy.spatial.distance import cdist
from typing import Tuple, Optional, Union, Dict, List

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.dielectric import DielectricProperties
from LayerModel_lib.tissue_properties import TissueProperties
from LayerModel_lib.coordinate import Coordinate


class FrequencyVector:

    _frequency_vector = None
    _endpoint = None
    _type = None

    def __init__(self, f_start: Optional[float] = 1e9,
                 f_end: Optional[float] = 10e9,
                 n_samples: Optional[int] = 2048,
                 frequency_vector: Union[str, np.ndarray] = 'natural',
                 endpoint: bool = True):
        """
        Generate a vector with frequencies in Hz from the given parameters.

        :param float f_start:       lower frequency
        :param float f_end:         higher frequency
        :param int n_samples:       number of frequencies bewteen f_start and f_end
        :param bool endpoint:       Determines whether f_end is included in the frequency vector or not.
        :param frequency_vector:    May be a numpy array containing all the frequency samples or a string. If it is a
                                    string the value determines how the frequencies between f_start and f_end are
                                    computed:
                                    if frequency_vector == 'natural' (frequency vector in natural ascending order)
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                    (A frequency vector that can directly be used with the np.fft.ifft function is generated)
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw) + fc

        :return A vector with frequencies in Hz
        """
        if isinstance(frequency_vector, np.ndarray):
            # the frequncy vector was given
            self._frequency_vector = frequency_vector
            self._frequency_vector.shape = (-1,)
        elif isinstance(frequency_vector, str):
            # generate the frequency values
            self._type = frequency_vector
            if frequency_vector == 'natural':
                # frequency vector in natural ascending order, including the endpoint depending on flag endpoint
                self._frequency_vector = np.linspace(f_start, f_end, n_samples, endpoint=endpoint)
                self._endpoint = endpoint
            elif frequency_vector == 'fft':
                # frequency vector that can directly be used with the np.fft.ifft function
                bw = (f_end - f_start)
                fc = f_start + bw / 2
                self._frequency_vector = np.fft.fftfreq(n_samples, 1/bw) + fc

        if self._frequency_vector is None:
            raise ValueError('Could not generate a frequency vector!')

    @property
    def start(self):
        return np.min(self._frequency_vector)

    @property
    def end(self):
        if self._type == 'fft':
            raise NotImplementedError
        else:
            return np.max(self._frequency_vector)

    @property
    def n_samples(self):
        return self._frequency_vector.size

    @property
    def bandwidth(self):
        return self.end - self.start

    @property
    def center(self):
        return self.start + self.bandwidth/2

    def __getitem__(self, item) -> np.ndarray:
        return self._frequency_vector[item]


class LayerModelBase:

    def __init__(self, voxel_model: VoxelModel = VoxelModel(),
                 startpoint: Coordinate = Coordinate([0, 0, 0]), endpoint: Coordinate = Coordinate([1, 1, 1]),
                 model_type: str = 'trunk',):
        """
        Initialize the base of the layer model.

        :param VoxelModel voxel_model:      A VoxelModel object that is used for the calculation of the layers
        :param numpy.ndarray startpoint:    Startpoint coordinate (transmitter location) inside the VoxelModel
        :param numpy.ndarray endpoint:      Endpoint coordinate (receiver location) outside the VoxelModel
        :param model_type:                  the type of the model to use, e.g. 'complete' or 'trunk'
        """
        self.startpoint = startpoint
        self.endpoint = endpoint
        self.vm = voxel_model
        self.model_type = model_type
        # the distance between transmitter and receiver
        self.distance = 0

    @staticmethod
    def frequency_vector(f_start: Optional[float] = 1e9,
                         f_end: Optional[float] = 10e9,
                         n_samples: Optional[int] = 2048,
                         frequency_vector: Union[str, np.ndarray] = 'natural',
                         endpoint: bool = False) -> Union[Tuple[np.ndarray, float], np.ndarray]:
        """
        Generate a vector with frequencies in Hz from the given parameters.

        :param float f_start:       lower frequency
        :param float f_end:         higher frequency
        :param int n_samples:       number of frequencies bewteen f_start and f_end
        :param bool endpoint:       Determines whether f_end is included in the frequency vector or not.
        :param frequency_vector:    May be a numpy array containing all the frequency samples or a string. If it is a
                                    string the value determines how the frequencies between f_start and f_end are
                                    computed:
                                    if frequency_vector == 'natural' (frequency vector in natural ascending order)
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                    (A frequency vector that can directly be used with the np.fft.ifft function is generated)
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw) + fc

        :return A vector with frequencies in Hz
        """
        frequencies = None
        fc = None
        if isinstance(frequency_vector, np.ndarray):
            # the frequncy vector was given
            frequencies = frequency_vector
            frequencies.shape = (-1,)
        elif isinstance(frequency_vector, str):
            # generate the frequency values
            if frequency_vector == 'natural':
                # frequency vector in natural ascending order, including the endpoint depending on flag endpoint
                frequencies = np.linspace(f_start, f_end, n_samples, endpoint=endpoint)
            elif frequency_vector == 'fft':
                # frequency vector that can directly be used with the np.fft.ifft function
                bw = (f_end - f_start)
                fc = f_start + bw / 2
                frequencies = np.fft.fftfreq(n_samples, 1/bw) + fc

        if frequencies is None:
            raise ValueError('Could not generate a frequency vector!')

        if fc is None:
            return frequencies
        else:
            return frequencies, fc

    def transfer_function(self, f_start: float = 1e9,
                          f_end: float = 10e9,
                          n_samples: int = 2048,
                          field_type: str = 'S21',
                          direction: str = 'start->end',
                          radiation_loss: str = None,
                          free_space_loss_distance: float = None,
                          frequency_vector: Union[str, np.ndarray] = 'natural',
                          endpoint: bool = False)\
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the transfer function at n equidistant frequency points from f_start to f_end.

        :param float f_start:      lower frequency
        :param float f_end:        higher frequency
        :param int n_samples:   number of evaluation points
        :param str field_type:  By default this is 'S21', can also be 'E' for electric field  or 'H' for magnetic field.
        :param direction:   sets the direction of the transfer function, either from startpoint to endpoint or
                            vice versa. Can either be 'start->end' or 'end->start'
        :param radiation_loss:  Select how to calculate the additional radiation loss that is added to the transfer
                                functions. Possible values are: 'air' (assume speed of light in free space) or
                                'avg_tau' (compute speed of light from group delay of transfer function).
        :param free_space_loss_distance: if not set to None the free space loss for the distance given here in m,
                                         will be included in the transfer function (works only for S21). Just
                                         kept for backwards compatibility.
        :param frequency_vector:    May be a numpy array containing all the frequency samples or a string. Then
                                    the value determines how the frequencies between f_start and f_end are computed.
                                    if frequency_vector == 'natural'
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw)

        :param endpoint: include f_end in the calculation (default False)
        :return:
        """
        raise NotImplementedError

    def path_loss(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                  precalculated_tf: Tuple = None,
                  transmit_psd: Union[str, np.ndarray] = 'const',
                  **kwargs) -> float:
        """
        Calculate the path loss in the frequency band between f_start and f_end using n_samples.
        It is assumed for this function that the transmit power spectral density is uniformly distributed
        between f_start and f_end. All **kwargs will be passed to self.transfer_function().

        :param float f_start:     lower frequency
        :param float f_end:       higher frequency
        :param int n_samples:   number of samples to use to evaluate the path loss
        :param Tuple precalculated_tf: A tuple containing (transfer_function, f) calculated already with
                                          self.transfer_function()
        :param np.ndarray transmit_psd Power spectral/energy density of the transmit signal in the same frequency range
                                       as the path loss is to be computed. Defaults to 'const',
                                       assuming a constant power/energy distribution over the complete bandwidth.
        :return: float: the path loss
        """

        if precalculated_tf is None:
            transfer_function, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (transfer_function, f) = precalculated_tf

        if isinstance(transmit_psd, str) and transmit_psd == 'const':
            bandwidth = f[-1] - f[0]
            path_gain = np.trapz(np.abs(transfer_function) ** 2, x=f, axis=0) / bandwidth
            path_loss = 1 / path_gain
        elif isinstance(transmit_psd, np.ndarray):
            input_power = np.trapz(transmit_psd, x=f, axis=0)
            output_power = np.trapz(transmit_psd * np.abs(transfer_function) ** 2, x=f, axis=0)
            path_loss = input_power / output_power
        else:
            raise ValueError("transmit_psd has to be either 'const' or a numpy array!")

        return path_loss

    def impulse_response(self, f_sample: float = 20e9,
                         n_samples: int = 2049,
                         f_start: float = None,
                         f_end: float = None,
                         normalize_energy: bool = False,
                         truncation_percentage: float = 1,
                         truncation: str = None,
                         normalize_delay: str = 'none',
                         threshold_dB: float = 30,
                         equivalent_baseband: Dict = None,
                         oversampling_factor: int = None,
                         window: Union[str, None] = 'hann',
                         transmit_psd: Union[str, np.ndarray] = 'const',
                         precalculated_tf: Tuple = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the impulse response with the given sampling rate or downconverted between
        frequncies f_start and f_end. All kwargs are passed into self.transfer_function().

        :param float f_sample:                The sampling rate of the impulse response.
        :param int n_samples:                 The number of samples of the underlying transfer function/ifft calculation
        :param bool normalize_energy:         Boolean to normalize the energy of the impulse response to 1
        :param float truncation_percentage:   Value in [0,1] to determine after which percentage of the energy the
                                              output gets truncated.
        :param str truncation:                Whether the impulse response should get truncated after some percentage of
                                              total energy ('percentage') or after crossing a threshold of
                                              threshold_dB ('threshold'). Defaults to None (no truncation)
        :param str normalize_delay:           Determines if the delay of the impulse response is kept ('none') or if the
                                              output is truncated such that the rising edge ('edge') or the
                                              peak ('peak') is at t=0
        :param float threshold_dB: threshold for determining the start of the impulse response if normalize_delay is
                                    set to 'edge'.
        :param equivalent_baseband: Used to convert the impulse response to equivalent baseband represantation.
                                    If this is not None it is assumed to be a dictionary with the following keys
                                    {'fc': carrier frequency of downconversion,
                                     'B': bandwidth for the equivalent baseband signal, centered at fc}.
                                    f_sample has no meaning in this case, as the new sampling rate is 2*B.
        :param Tuple precalculated_tf: A tuple containing (transfer_function, f) calculated already with
                                          self.transfer_function()

        :return: impulse response, t:  The impulse response and corresponding time vector t
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        downconversion = False
        # No downconversion at all. Compute impulse response for 0 < f < fs/2
        if equivalent_baseband is None and (f_start is None or f_end is None) and f_sample is not None:
            f_start = 0
            f_end = f_sample/2
            bandwidth = f_end
            # if no downconversion is taking place we need to half the number of samples to get the desired number of
            # samples at the output (only the transfer function for the positive frequencies is generated
            n_samples = int(n_samples/2+1)
            # the same holds for the transmit_psd from that we need only the positive frequencies
            transmit_psd = transmit_psd[0: int(transmit_psd.size / 2 + 1)]

        # Downconversion using equivalent_baseband dict
        else:
            downconversion = True
            if equivalent_baseband is not None and f_start is None and f_end is None:
                # define variables for downconversion
                bandwidth = equivalent_baseband['B']
                carrier_frequency = equivalent_baseband['fc']
                f_start = carrier_frequency - bandwidth/2
                f_end = carrier_frequency + bandwidth/2
                f_sample = bandwidth
                f_BB = np.linspace(-bandwidth/2, bandwidth/2, n_samples)  # frequencies in eq. baseband
            # Downconversion using start and end frequencies
            elif f_start is not None and f_end is not None and equivalent_baseband is None:
                bandwidth = f_end - f_start
                f_sample = bandwidth
            else:
                raise ValueError("Invalid combination of equivalent_baseband, f_start, f_end and f_sample provided.")

        if precalculated_tf is None:
            transfer_function, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (transfer_function, f) = precalculated_tf

        # Apply windowing
        if window is not None:
            if window in ["hamming", "hann", "blackman"]:
                window_map = {'hamming': 'hamming', 'hann': 'hanning', 'blackman': 'blackman'}

                # when a window function is applied the total energy is reduced
                # thus we need to scale the transfer function
                pathloss_tf_before_window = self.path_loss(precalculated_tf=(transfer_function, f),
                                                           transmit_psd=transmit_psd)
                logging.debug('before window %2.2f dB' % (10 * np.log10(pathloss_tf_before_window)))

                if downconversion:
                    w = getattr(np, window_map[window])(transfer_function.size)
                    w.shape = (-1,)
                    transfer_function = w * transfer_function
                else:
                    # if no downconversion is applied, transfer function contains only the positive frequencies
                    # -> hence only half the window has to be applied
                    w = getattr(np, window_map[window])((transfer_function.size-1)*2)
                    w.shape = (-1,)
                    transfer_function = w[transfer_function.size-2::] * transfer_function

                pathloss_tf_after_window = self.path_loss(precalculated_tf=(transfer_function, f),
                                                          transmit_psd=transmit_psd)
                logging.debug('after window %2.2f dB' % (10 * np.log10(pathloss_tf_after_window)))

                scaling_factor = np.sqrt(pathloss_tf_after_window / pathloss_tf_before_window)
                logging.debug('scaling factor %e' % scaling_factor)

                transfer_function = scaling_factor * transfer_function

                pathloss_tf_final = self.path_loss(precalculated_tf=(transfer_function, f),
                                                   transmit_psd=transmit_psd)
                logging.debug('final %2.2f dB' % (10 * np.log10(pathloss_tf_final)))
            else:
                logging.warning(f'Unsupported window {window}. Fallback to using no window at all.')

        # allow for oversampling for smoother plots -> zero padding of the transfer function
        if oversampling_factor is not None and oversampling_factor > 1:
            f_sample = oversampling_factor * f_sample
            N = transfer_function.size
            transfer_function.shape = (-1,)
            transfer_function = np.pad(transfer_function,
                                       pad_width=int((oversampling_factor - 1) * N / 2),
                                       mode='constant')

        # make sure that transfer_function is a column vector
        transfer_function.shape = (-1, 1)

        # select the correct IFFT function for transformation
        if not downconversion:
            # Take the real IFFT irfft, as we want the output to be real and assume the spectrum to be Hermitian
            # as long as we compute the impulse response in the passband
            ifft = lambda x: np.fft.irfft(x, axis=0) * f_sample
            # impulse_response = np.fft.irfft(transfer_function, axis=0)
        else:
            # For equivalent baseband we need to take the IFFT, but first reorder the transfer function, such that the
            # zero frequency component is in transfer_function[0].
            transfer_function = np.fft.ifftshift(transfer_function)
            ifft = lambda x: np.fft.ifft(x, axis=0) * f_sample
            # impulse_response = np.fft.ifft(tf_BB, axis=0)

        # compute the impulse response:
        impulse_response = ifft(transfer_function)

        # Total energy of impulse response
        dt = 1/f_sample
        energy = np.sum(np.abs(impulse_response) ** 2, axis=0) * dt

        # normalize energy of impulse response to 1
        if normalize_energy:
            impulse_response = impulse_response / np.sqrt(energy)
            energy = 1

        if truncation == 'percentage':
            # If the impulse response is supposed to be truncated after some percentage of energy
            # truncation_percentage. If truncation_percentage >= 1 no truncation is done
            if truncation_percentage < 1:
                k_max = 0
                for k in range(0, impulse_response.shape[0]):
                    energy_k = np.sum(np.abs(impulse_response[0:k, :]) ** 2, axis=0) * dt
                    if energy_k > energy * truncation_percentage:
                        k_max = k
                        break

                if k_max != 0:
                    impulse_response = impulse_response[0:k_max, :]
        elif truncation == 'threshold':
            # truncate the impulse response to the length after which the instantaneous energy is smaller than
            # threshold_dB compared to the peak
            peak_index = np.argmax(np.abs(impulse_response))
            threshold_value = np.abs(impulse_response[peak_index]) * 10 ** (- threshold_dB / 20)
            # find the index where the threshold is first met after the peak of the impulse response
            threshold_index = np.flatnonzero(np.abs(impulse_response[peak_index::]) < threshold_value)[0]
            # truncate the impulse response after the threshold has been crossed
            impulse_response = impulse_response[0:peak_index+threshold_index]

        # Normalize the delay of the pulse either such that
        #   the peak is at t=0, ('peak')
        #   the start of the rising edge ('edge') or
        #   do not normalize at all ('none')
        if normalize_delay == 'peak':
            start_index = np.argmax(abs(impulse_response), axis=0)
            impulse_response = impulse_response[start_index[0]:]
        elif normalize_delay == 'edge':
            # determine the value of the maximum - threshold_dB (amplitudes -> factor 20!)
            threshold_value = np.max(np.abs(impulse_response)) * 10 ** (- threshold_dB / 20)
            # find the index where the threshold is first met
            threshold_index = np.argmax(np.abs(impulse_response) > threshold_value)
            if threshold_index == impulse_response.size - 1:
                start_index = 0
            else:
                start_index = threshold_index

            impulse_response = impulse_response[start_index:]
        elif normalize_delay != 'none':
            raise ValueError('normalize_time can either be "peak", "edge", or "none"')

        # Generate time vector
        t = np.linspace(0, impulse_response.size / f_sample, impulse_response.size, endpoint=False)
        t.shape = (-1,)
        impulse_response.shape = (-1,)
        return impulse_response, t

    def group_delay(self, average: bool = True,
                    f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                    precalculated_tf: Tuple = None, **kwargs) -> Union[np.ndarray, float]:
        """
        Compute the group delay of the layer model in a certain frequency range.
        All **kwargs will be passed to self.transfer_function().

        :param average: average the group delay over all frequencies (default)
        :param float f_start:     lower frequency
        :param float f_end:       higher frequency
        :param int n_samples:   number of samples to use to evaluate the path loss
        :param Tuple precalculated_tf: A tuple containing (transfer_function, f) calculated already with
                                          self.transfer_function()
        :return:
        """
        if precalculated_tf is None:
            transfer_function, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (transfer_function, f) = precalculated_tf

        # determine speed of light by calculating average speed of light from group delay
        tau_g = - np.gradient(np.unwrap(np.angle(transfer_function.flat)), f.flat) / (2 * np.pi)

        if average:
            tau_g = np.mean(tau_g)

        return tau_g

    def propagation_delay(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                          frequency_vector: Union[str, np.ndarray] = 'natural',
                          endpoint: bool = True) -> Union[Dict, np.ndarray]:
        """
        Compute the propagation delay through the layer model for a given frequency range. All parameters are
        handed to self.frequency_vector().

        :return:
        """
        raise NotImplementedError

    def phase_velocity(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                       frequency_vector: Union[str, np.ndarray] = 'natural',
                       endpoint: bool = True) -> Union[Dict, np.ndarray]:
        """
        Compute the phase velocity through the layer model for a given frequency range. All parameters are
        handed to self.frequency_vector(). Returns an array with the phase velocity for each frequency.

        :return:
        """
        raise NotImplementedError

    def channel_capacity(self, noise_power_density: float, transmit_power: float,
                         f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                         precalculated_tf: Tuple = None, **kwargs) -> float:
        """
        Calculate the channel capacity for a real channel (e.g. UWB) for a rectangular PSD of the transmit signal
        in the range of f_start to f_end calculated using n_samples.
        All **kwargs will be passed to self.transfer_function().

        :param noise_power_density: The power spectral density of the noise
        :param float transmit_power: the total transmit power that is then equally distributed in the bandwidth
                                     from f_start to f_end
        :param float f_start: lower frequency limit
        :param float f_end: upper frequency limit
        :param int n_samples: resolution of the calculation of the transfer function
        :param Tuple precalculated_tf: A tuple containing (S21, f) calculated already with self.S21()
        :return:
        """
        if precalculated_tf is None:
            tf, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (tf, f) = precalculated_tf
            f_start = f[0]
            f_end = f[-1]

        bandwidth = f_end - f_start

        path_gain = np.abs(tf) ** 2

        # the energy per symbol is the magnitude of the power spectral density
        energy_per_symbol = transmit_power / (2 * bandwidth)
        transmit_power_density = energy_per_symbol * np.ones(tf.shape)

        capacity = np.trapz(np.log2(1 + 2 * (transmit_power_density * path_gain) / noise_power_density), x=f, axis=0)

        return capacity


class DirectLinkLayerModel(LayerModelBase):
    """"
    The class describing one communication link inside the human body.

    Main Properties are:
    startpoint                      The location of the transmit antenna inside the human body
    endpoint                        The location of the receive antenna on the body surface
    tissue_array                    Contains all tissues from start to end
                                    (as index in the TissueProperties.tissue_names)
    depth_array                     The thickness of each tissue layer in tissue_array
    source_tissue                   Tissue type at the startpoint
    source_impedance_tissue_array   Tissue types used to calculate the source impedance
    source_impedance_depth_array    The thickness of each tissue layer in source_impedance_tissue_array
    source_impedance_load_tissue    The tissue that is used for the load impedance of the source impedance calculation
    """

    def __init__(self, voxel_model: VoxelModel = VoxelModel(),
                 startpoint: Coordinate = Coordinate([0, 0, 0]), endpoint: Coordinate = Coordinate([1, 1, 1]),
                 tissue_properties: DielectricProperties = TissueProperties(),
                 model_type: str = 'trunk', layermodel: 'DirectLinkLayerModel' = None):
        """
        Generate the layer model for voxel_model

        :param TissueProperties tissue_properties:  Object containing the dielectric tissue properties
        :param VoxelModel voxel_model:              A VoxelModel object that is used for the calculation of the layers
        :param numpy.ndarray startpoint:            Startpoint coordinate (transmitter location) inside the VoxelModel
        :param numpy.ndarray endpoint:              Endpoint coordinate (receiver location) outside the VoxelModel
        :param model_type:   the type of the model to use, e.g. 'complete' or 'trunk'
        """
        if layermodel is not None:
            # import the data from an already existing layer model into this instance of the class
            self.__dict__.update(layermodel.__dict__)
        else:
            # The tissue property object is used to calculate the dielectric properties
            self.tissue_properties = tissue_properties  # TissueProperties()
            # the arrays that hold the tissues and their thickness in the end
            self.tissue_array = np.zeros(shape=(0,))
            self.depth_array = np.zeros(shape=(0,))

            # by default we set the source medium to a layer of 1000mm air
            self.source_tissue = 0
            self.source_impedance_tissue_array = np.array([0])
            self.source_impedance_depth_array = np.array([[1000]]) * 1e-3
            self.source_impedance_load_tissue = np.array([0])
            # the distance between transmitter and receiver
            self.distance = 0

            super().__init__(voxel_model, startpoint, endpoint, model_type)

            # check if the voxel_model is empty
            # only get the tissue layers from the model if it is not empty
            # otherwise the tissue layers need be set individually by hand
            if voxel_model.name != 'empty':
                self.get_tissues_from_voxel_model(voxel_model, startpoint, endpoint, model_type=model_type)

    def get_tissues_from_voxel_model(self, voxel_model: VoxelModel=VoxelModel(),
                                     startpoint: Coordinate=Coordinate([0, 0, 0]),
                                     endpoint: Coordinate=Coordinate([1, 1, 1]),
                                     model_type: str='trunk'):
        """
        Determine the tissues between start and endpoint in self.voxel_model
        :param voxel_model: The VoxelModel to investigate
        :param startpoint: the startpoint
        :param endpoint: the endpoint
        :param model_type: the type of the model to use, e.g. 'complete' or 'trunk'
        :return:
        """
        # Find the tissues between startpoint_mm and endpoint_mm
        tissues = voxel_model.tissue_finding_3D(startpoint, endpoint, model_type=model_type)
        self.tissue_array = tissues['tissue_array']
        self.depth_array = tissues['depth_array']
        self.endpoint = tissues['endpoint']
        self.source_tissue = tissues['source_tissue']

        # Find the tissues needed for the source impedance calculation
        # For calculation of the source impedance we need the tissue layers in the opposite direction for about 100 mm.
        # Due to the heavy attenuation this should be sufficient.
        # As the direction of the line in tissue_finding is calculated based on the voxel indices do the same here:
        startpoint_d = np.around(startpoint / voxel_model.scaling).astype(int)
        endpoint_d = np.around(endpoint / voxel_model.scaling).astype(int)
        # direction in voxel indices:
        direction_d = (endpoint_d - startpoint_d)
        # direction in coordinates
        direction = direction_d * voxel_model.scaling / np.linalg.norm(direction_d * voxel_model.scaling)

        # it can happen that the voxel 100mm in the opposite direction is outside the model.
        # for that reason check if it is inside (by a test conversion to index) and otherwise reduce the distance
        distance = 100  # mm
        while distance >= 0:
            # calculate the endpoint for the source impedance calculation
            source_endpoint = startpoint - distance * direction  # 100 mm in opposite direction
            try:
                voxel_model.models[model_type].coordinate_to_index(source_endpoint, voxel_model.scaling)
                break
            except IndexError:
                distance -= 5
                logging.debug("source_endpoint = %s is outside model! "
                              "Distance reduced from 100 mm to %d (startpoint = %s, endpoint = %s) "
                              % (str(source_endpoint),  distance, str(startpoint), str(endpoint)))

        if distance < 100:
            logging.debug("Distance for source impedance calculation was reduced fromm 100 mm to %d mm" % distance)

        tissues = voxel_model.tissue_finding_3D(startpoint, source_endpoint,
                                                return_load_tissue=True, model_type=model_type)
        self.source_impedance_tissue_array = tissues['tissue_array']
        self.source_impedance_depth_array = tissues['depth_array']
        self.source_impedance_load_tissue = tissues['load_tissue']

        # Calculate the distance between transmitter and receiver
        self.distance = np.sum(self.depth_array)

    def print_info(self):
        """
        Print an info string containing all relevant information to this layer model.
        """

        total_distance = self.distance * 1e3
        print("========= DirectLinkLayerModel info =========\n\n"
              "* Start: \t[%.2f %.2f %.2f]" % (self.startpoint.x, self.startpoint.y, self.startpoint.z) +
              "\n* End: \t\t[%.2f %.2f %.2f]" % (self.endpoint.x, self.endpoint.y, self.endpoint.z) + "\n"
              "* Distance: \t%.2f mm" % total_distance + "\n"
              "* Layer: \n"
              "--------")

        print("--------\n %s (\u27f6\u221e)" % self.tissue_properties.get_name_for_id(self.source_impedance_load_tissue)[0])

        source_tissues = self.tissue_properties.get_name_for_id(self.source_impedance_tissue_array)
        if len(source_tissues) == 0:
            print("------------------------ TX Location")
        else:
            print("------------------------")
            for k in range(len(source_tissues)-1, -1, -1):
                d = self.source_impedance_depth_array[k] * 1e3
                output_string = source_tissues[k] + " (%.2f mm) \n------------------------" % d
                if k == 0:
                    end_string = ' TX Location\n'
                else:
                    end_string = '\n'
                print(output_string, end=end_string)

        tissues = self.tissue_properties.get_name_for_id(self.tissue_array)
        for k in range(0, len(tissues)):
            d = self.depth_array[k]*1e3
            output_string = tissues[k] + " (%.2f mm) \n------------------------" % d
            if k == len(tissues)-1:
                end_string = ' RX Location\n'
            else:
                end_string = '\n'
            print(output_string, end=end_string)

        print(" Air (\u27f6\u221e)\n--------")
        print("\n========== End =========")

    def plot(self, title: str=None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the tissue layers in a new figure.

        :return: The figure and axes handle of the plot.
        """
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Rectangle

        hf, ha = plt.subplots()
        # list for all the layer rectangles
        layers = []
        total_distance = self.distance * 1e3
        ha.set_title("From [%.2f %.2f %.2f]" % (self.startpoint.x, self.startpoint.y, self.startpoint.z) +
                     "To [%.2f %.2f %.2f]" % (self.endpoint.x, self.endpoint.y, self.endpoint.z) + "\n")

        # add source impedance tissue:
        source_impedance_load_tissue = self.tissue_properties.get_name_for_id(self.source_impedance_load_tissue)[0]
        source_impedance_thickness = np.sum(self.source_impedance_depth_array) * 1e3
        last_x = -1e3
        d = np.abs(last_x) - source_impedance_thickness
        pc = PatchCollection([Rectangle((last_x, 0), d, 1)],
                             facecolor=self.vm.colormap.colors[int(self.source_impedance_load_tissue), :],
                             edgecolors='black')
        ha.add_collection(pc)
        last_x = last_x + d
        ha.text(x=last_x - 5, y=0.2, s=source_impedance_load_tissue, rotation='vertical', verticalalignment='top',
                horizontalalignment='center')

        # loop over all source impedance tissues:
        source_tissues = self.tissue_properties.get_name_for_id(self.source_impedance_tissue_array)
        for k in range(len(source_tissues) - 1, -1, -1):
            d = self.source_impedance_depth_array[k] * 1e3
            pc = PatchCollection([Rectangle((last_x, 0), d, 1)],
                                 facecolor=self.vm.colormap.colors[int(self.source_impedance_tissue_array[k]), :],
                                 edgecolors='black')
            ha.add_collection(pc)
            ha.text(x=last_x + d/2, y=0.2, s=source_tissues[k], rotation='vertical', verticalalignment='top',
                    horizontalalignment='center')
            last_x = last_x + d

        # loop over all tissue layers
        tissues = self.tissue_properties.get_name_for_id(self.tissue_array)
        for (k, t) in enumerate(tissues):
            d = self.depth_array[k] * 1e3
            pc = PatchCollection([Rectangle((last_x, 0), d, 1)],
                                 facecolor=self.vm.colormap.colors[int(self.tissue_array[k]), :],
                                 edgecolors='black')
            ha.add_collection(pc)
            ha.text(x=last_x + d/2, y=0.2, s=t, rotation='vertical', verticalalignment='top',
                    horizontalalignment='center')
            last_x = last_x + d

        ha.set_xlim([-source_impedance_thickness*1.1, total_distance*1.1])
        ha.set_ylim([0, 1])
        ha.set_xlabel('Distance in mm')
        if title is not None:
            ha.set_title(title)

        return hf, ha

    def t_matrix(self, tissue_array: np.ndarray, depth_array: np.ndarray, f: Union[np.ndarray, float])\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the transmission parameter matrix for the tissue types in tissue_array with their corresponding depth
        found in depth_array. The matrix is calculated for each value in f.

        :param numpy.ndarray tissue_array: A vector containing all the tissue indices of the model
        :param numpy.ndarray depth_array: The thickness of the respective layers in the model.
        :param numpy.ndarray f: The frequencies for which the T matrix is to be computed.
        :return: numpy.ndarray : a tuple of the transmission parameters (A, B, C, D), where each element is a vector
                                 over all frequencies.
        """
        # make sure f is column vector and d is a row vector
        # but only do that if they are ndarray. For one value, especially, when calculating source impedance,
        # depth_array might be a float.
        if type(f) is not np.ndarray:
            f = np.array([f])

        f.shape = (-1, 1)
        lf = f.shape[0]

        depth_array = np.array([depth_array])
        if type(tissue_array) is not np.ndarray:
            # if it is not an ndarray, cast depth_array to ndarray and tissue_array to list
            tissue_array = np.array(tissue_array, dtype=np.uint8)

        if len(tissue_array) == 0:
            # if there are no tissues, T-matrix is identity matrix
            A = np.array([1])
            B = np.array([0])
            C = np.array([0])
            D = np.array([1])
            return A, B, C, D
        else:
            depth_array.shape = (1, -1)
            # length of the dimensions
            ld = depth_array.shape[1]

            # repeat the depth array for each frequency
            d = np.tile(depth_array, (lf, 1))

            # For the following computations: rows have same frequency and columns are from the same layer. Example:
            # 1(air) 1(skin) 1(fat) 1(muscle) 1(fat) GHz
            # 2(air) 2(skin) 2(fat) 2(muscle) 2(fat) GHz

            # calculate eta and gamma for all layers and frequencies
            eta = self.tissue_properties.wave_impedance(tissue_array, f)
            gamma = self.tissue_properties.propagation_constant(tissue_array, f)

            # calculate the T Matrix
            psi = gamma * d
            At = np.cosh(psi)
            Bt = eta * np.sinh(psi)
            Ct = 1 / eta * np.sinh(psi)
            Dt = At  # np.cosh(psi)

            # Resort the dimensions:
            # f is third dimension
            # layer the fourth
            T = np.zeros((2, 2, lf, ld), dtype=np.complex128)
            T[0, 0, :, :] = At
            T[0, 1, :, :] = Bt
            T[1, 0, :, :] = Ct
            T[1, 1, :, :] = Dt

            # Multiply the T-matrices of the layers with each other. This is done for each frequency separately
            t = np.eye(2, 2)
            t.shape = (2, 2, 1)
            Ttotal = np.tile(t, (1, 1, lf))
            for layer in range(0, ld):
                Ttotal = np.einsum('ij...,jk...->ik...', Ttotal, T[:, :, :, layer])
                # this is basically a matrix-matrix product of the first two dimensions of T leaving the third
                # (the frequency as it is) .

            # Return the elements of the T-Matrix separately, remove singleton dimensions and make it a column vector
            A = np.squeeze(Ttotal[0, 0, :])
            A.shape = (-1, 1)
            B = np.squeeze(Ttotal[0, 1, :])
            B.shape = (-1, 1)
            C = np.squeeze(Ttotal[1, 0, :])
            C.shape = (-1, 1)
            D = np.squeeze(Ttotal[1, 1, :])
            D.shape = (-1, 1)

            return A, B, C, D

    def source_impedance(self, f: np.ndarray) -> np.ndarray:
        """
        Calculate the source impedance of the model.

        :param numpy.ndarray f: frequency vector
        :return: numpy.ndarray: the source impedance for each frequency
        """
        # replace zero entries with 100 Hz for calculation
        f_i = np.nonzero(f == 0)
        f[f_i] = 100

        # load impedance of the source impedance calculation
        source_eta_L = self.tissue_properties.wave_impedance(self.source_impedance_load_tissue, f)
        # calculate the T matrix in source impedance direction
        (A, B, C, D) = self.t_matrix(self.source_impedance_tissue_array, self.source_impedance_depth_array, f)
        eta_S = (source_eta_L * A + B) / (source_eta_L * C + D)

        return eta_S

    def transfer_function(self,
                          f_start: float = 1e9,
                          f_end: float = 10e9,
                          n_samples: int = 2048,
                          field_type: str = 'S21',
                          direction: str = 'start->end',
                          radiation_loss: str = None,
                          free_space_loss_distance: float = None,
                          frequency_vector: Union[str, np.ndarray] = 'natural',
                          endpoint: bool = False)\
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the transfer function at n equidistant frequency points from f_start to f_end.
        :param float f_start:      lower frequency
        :param float f_end:        higher frequency
        :param int n_samples:   number of evaluation points
        :param str field_type:  By default this is 'S21', can also be 'E' for electric field  or 'H' for magnetic field.
        :param direction:   sets the direction of the transfer function, either from startpoint to endpoint or
                            vice versa. Can either be 'start->end' or 'end->start'
        :param radiation_loss:  Select how to calculate the additional radiation loss that is added to the transfer
                                functions. Possible values are: 'air' (assume speed of light in free space) or
                                'effective_tissue' (compute phase velocity from effective propagation delay in the
                                tissue layers).
        :param free_space_loss_distance: if not set to None the free space loss for the distance given here in m,
                                         will be included in the transfer function (works only for S21). Just
                                         kept for backwards compatibility.
        :param frequency_vector:    May be a numpy array containing all the frequency samples or a string. Then
                                    the value determines how the frequencies between f_start and f_end are computed.
                                    if frequency_vector == 'natural'
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw) + fc

        :param endpoint: include f_end in the calculation (default False)
        :return:
        """
        if isinstance(frequency_vector, str) and frequency_vector == 'fft':
            f_theoretic, fc = self.frequency_vector(f_start, f_end, n_samples, frequency_vector, endpoint)
        else:
            f_theoretic = self.frequency_vector(f_start, f_end, n_samples, frequency_vector, endpoint)

        depth_array = np.array([self.depth_array])
        depth_array.shape = (1, -1)

        # if f_start is zero raise an exception as this is not supported.
        if f_start == 0:
            raise ValueError('The lower frequency for transfer function computation must not be 0 Hz.')
        else:
            f = f_theoretic

        if direction == 'start->end':
            # First determine the load impedance. which is assumed to be Air
            eta_L = self.tissue_properties.wave_impedance(self.tissue_properties.get_id_for_name('Air'), f[0])

            # Second the Source impedance
            eta_S = self.source_impedance(f)

            # Determine the transmission parameters of the layer
            (A, B, C, D) = self.t_matrix(self.tissue_array, depth_array, f)
        elif direction == 'end->start':
            # First determine the load impedance. This is the source impedance of the default setup
            eta_L = self.source_impedance(f)

            # Second the Source impedance, this is air in this case
            eta_S = self.tissue_properties.wave_impedance(self.tissue_properties.get_id_for_name('Air'), f[0])

            # Determine the transmission parameters of the layer
            # tissue_array and depth_array need to be flipped to reflect the inverted transmission direction
            # flipud and fliplr do the same thing in this case (the dimensions are just different...)
            (A, B, C, D) = self.t_matrix(np.flipud(self.tissue_array),
                                         np.fliplr(depth_array), f)
        else:
            raise ValueError("direction has to be either 'start->end' or 'end->start'.")

        if field_type == 'E':
            transfer_function = (2 * eta_L) / \
                                (eta_L * A + B + eta_S * eta_L * C + eta_S * D)
        elif field_type == 'H':
            transfer_function = (2 * eta_S) / \
                                (eta_L * A + B + eta_S * eta_L * C + eta_S * D)
        elif field_type == 'S21':
            if direction == 'end->start':
                raise ValueError("Direction can only be 'start->end' for computation of S21.")
            transfer_function = (np.sqrt(np.real(eta_L)) * 2 * eta_S) / \
                                (np.sqrt(np.real(eta_S)) * (eta_L * A + B + eta_S * eta_L * C + eta_S * D))

            transfer_function.shape = (-1,)
            rl = self.radiation_loss(type=radiation_loss, f_start=f_start, f_end=f_end, n_samples=n_samples,
                                     frequency_vector=frequency_vector, endpoint=endpoint)

            transfer_function = transfer_function / rl
        else:
            raise ValueError("'field_type' can only be 'S21', 'E' or 'H'")

        # scale f_theoretic to baseband if frequency_vector == 'fft'
        if isinstance(frequency_vector, str) and frequency_vector == 'fft':
            f_theoretic = f_theoretic - fc

        # return only 1D array
        transfer_function.shape = (-1,)
        f_theoretic.shape = (-1,)

        return transfer_function, f_theoretic

    def S21(self, f_start: float = 0.0, f_end: float = 10e9, n_samples: int = 2048,
            frequency_vector: str = 'natural', endpoint: bool = False )\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate S21 for this layer model at n equidistant frequency points from f_start to f_end.
        Kept only for backwards compatibility (replaced by transfer_function()).

        :param float f_start:      lower frequency
        :param float f_end:        higher frequency
        :param int n_samples:   number of evaluation points
        :param frequency_vector: A string determining how the frequencies between f_start and f_end are computed.
                                    if frequency_vector == 'natural'
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw)

        :param endpoint: include f_end in the calculation (default False)

        :return:  numpy.ndarray    S21 for this layer model and the corresponding frequency vector (S21, f)
        """
        # generate the frequency values
        if frequency_vector == 'natural':
            # frequency vector in natural ascending order, including the endpoint depending on flag endpoint
            f_theoretic = np.linspace(f_start, f_end, n_samples, endpoint=endpoint)
        elif frequency_vector == 'fft':
            # frequency vector that can directly be used with the np.fft.ifft function
            bw = (f_end - f_start)
            fc = f_start + bw / 2
            f_theoretic = np.fft.fftfreq(n_samples, 1 / bw) + fc

        f_theoretic.shape = (-1, 1)

        # if f_start is zero this needs to be replaced by 10 Hz for calculation
        if f_start == 0:
            f = np.vstack([10, f_theoretic[1:]])
        else:
            f = f_theoretic

        # First determine the load impedance, which is assumed to be Air
        eta_L = self.tissue_properties.wave_impedance(np.array(self.tissue_properties.get_id_for_name('Air')),
                                                      np.array(f[0]))

        # Second the Source impedance
        eta_S = self.source_impedance(f)

        # Determine the transmission parameters of the layer
        (A, B, C, D) = self.t_matrix(self.tissue_array, self.depth_array, f)

        S21 = (np.sqrt(np.real(eta_L)) * 2 * eta_S) / \
              (np.sqrt(np.real(eta_S)) * (eta_L * A + B + eta_S * eta_L * C + eta_S * D))

        # return only 1D array
        S21.shape = (-1,)
        f_theoretic.shape = (-1,)
        return S21, f_theoretic

    def propagation_delay(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                          frequency_vector: Union[str, np.ndarray] = 'natural', endpoint: bool = True) -> np.ndarray:
        """
        Compute the propagation delay through the layer model for a given frequency range. All parameters are
        handed to self.frequency_vector(). Returns an array with the propagation delay for each frequency.

        :return:
        """
        f = self.frequency_vector(f_start, f_end, n_samples, frequency_vector, endpoint)
        f_orig_shape = f.shape
        # depth and frequency need to be 2D to vectorize the computation
        f.shape = (-1, 1)
        lf = f.shape[0]

        depth_array = np.copy(self.depth_array)
        depth_array.shape = (1, -1)

        # repeat the depth array for each frequency
        distance = np.tile(depth_array, (lf, 1))

        # calculate the propagation velocity (phase velocity) for each frequency.
        gamma = self.tissue_properties.propagation_constant(self.tissue_array, f)
        phase_velocity = 2 * np.pi * f / np.imag(gamma)
        # propagation delay for each layer
        tau = distance / phase_velocity
        # total propagation delay / each column holds one tissue layer
        tau_ges = np.sum(tau, axis=1)

        # restore original shape of frequency vector
        f.shape = f_orig_shape

        return tau_ges

    def phase_velocity(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                       frequency_vector: Union[str, np.ndarray] = 'natural', endpoint: bool = True) -> np.ndarray:
        """
        Compute the phase velocity through the layer model for a given frequency range. All parameters are
        handed to self.frequency_vector(). Returns an array with the phase velocity for each frequency.

        :return:
        """
        tau_ges = self.propagation_delay(f_start, f_end, n_samples, frequency_vector, endpoint)

        return self.distance / tau_ges

    def radiation_loss(self, type: Union[str, None], f_start: float = 3.1e9, f_end: float = 4.8e9,
                       n_samples: int = 1024, frequency_vector: Union[str, np.ndarray] = 'natural',
                       endpoint: bool = False) -> Union[np.ndarray, float]:
        """
        Compute the radiation loss due to spherical radiation of the wave. The loss is either based on the
        phase velocity in air (type='air') or based on the effective phase velocity in the layer model
        (type='effective_tissue'). All parameters except 'type' will be forwarded to self.frequency_vector()

        :param type: either 'air',
                            'effective_tissue' ('avg_tau' maps to 'effective_tissue' for backwards compatibility)
                            or None
        :param frequency_vector:  a numpy array, 'natural' or 'fft'
        :param float f_start:     lower frequency
        :param float f_end:       higher frequency
        :param int n_samples:   number of samples to use to evaluate the path loss

        :return:
        """
        if type not in ['air', 'effective_tissue', 'avg_tau', None, 'none', 'None']:
            raise ValueError("Type can be either 'air', 'effective_tissue', 'avg_tau' or None.")

        f = self.frequency_vector(f_start=f_start, f_end=f_end, n_samples=n_samples,
                                  frequency_vector=frequency_vector, endpoint=endpoint)

        c0 = 299792458  # m/s in free space!
        c = None
        d = self.distance

        if type == 'air':
            c = c0

        elif type == 'effective_tissue' or type == 'avg_tau':
            # determine phase velocity through the layers
            c = self.phase_velocity(frequency_vector=f)

        # c is only not None if some additional losses need to be added.
        if c is not None:
            wavelength = c / f
            return (4 * np.pi * d) / wavelength
        else:
            return 1

    @staticmethod
    def create_from_dict(layers: list, tissue_properties: DielectricProperties = TissueProperties()) \
            -> 'DirectLinkLayerModel':
        """
        Create a layer model from the data specified in the two dicts layer and source_layer.

        This function needs Python > 3.6 as it relies on the fact the a dict is stored in the order it was defined.
        See https://docs.python.org/3/whatsnew/3.6.html#whatsnew36-compactdict for details.

        :param layers:   A list containing a dictionary for each layer.
                        The  tissue name as key and the thickness of that tissue in mm as value for each dictionary.
                        An example list: [{'Air': None}, {'Fat': 200}, {'TX': None}, {'Muscle': 10}, {'RX': None}]
                          - The first entry is the tissue that is used for the load impedance calculation of the
                            source impedance
                          - Then all tissues between the first and the TX location appear.
                          - The TX position in the layer setup is marked with a 'TX': None.
                          - After the TX the layers between TX and RX are given.
                          - The last entry should be 'RX': None for completeness.
                        The example dict above results in the following DirectLinkLayerModel:
                            --------
                             Air (⟶∞)
                            ------------------------
                            Fat (200.00 mm)
                            ------------------------ TX Location
                            Muscle (10.00 mm)
                            ------------------------ RX Location
                             Air (⟶∞)
                            --------
        :param tissue_properties: The TissueProperties to use.

        :return:
        """
        lm = DirectLinkLayerModel(tissue_properties=tissue_properties)

        # Read in all the tissue layers between TX and RX
        layer_list = []
        source_layer_list = []
        depth_list_mm = []
        source_depth_list_mm = []
        first = True  # first run of for loop
        adding_source_tissues = True  # the dict always starts with the source tissues
        found_receiver = False
        for layer in layers:
            for (tissue, thickness) in layer.items():

                # first entry is the load impedance for the source impedance calculation
                if first:
                    if thickness is not None:
                        raise ValueError('The thickness of the first entry in layer has to be None.')
                    else:
                        lm.source_impedance_load_tissue = np.array(tissue_properties.get_id_for_name(tissue))
                        first = False
                        continue
                else:
                    # encountered TX position source tissues finished
                    if tissue == 'TX':
                        if thickness is None:
                            adding_source_tissues = False
                            continue
                        else:
                            raise ValueError('The thickness of the "TX" entry in layer has to be None.')

                    # RX reached break the loop
                    if tissue == 'RX':
                        if thickness is None:
                            found_receiver = True
                            break
                        else:
                            raise ValueError('The thickness of the "RX" entry in layer has to be None.')

                    # add the tissues to their respective list
                    if adding_source_tissues:
                        source_layer_list.append(tissue)
                        source_depth_list_mm.append(thickness)
                    else:
                        layer_list.append(tissue)
                        depth_list_mm.append(thickness)

            if found_receiver:
                break
        else:
            logging.warning("layer did not contain an 'RX' entry at the end. It should for completeness.")

        lm.tissue_array = np.array(tissue_properties.get_id_for_name(layer_list))
        lm.depth_array = np.array([depth_list_mm]).T * 1e-3

        # both lists need to be reversed as lm.source_impedance_tissue_array expects the list starting from TX
        lm.source_impedance_tissue_array = np.array(tissue_properties.get_id_for_name(source_layer_list[::-1]))
        lm.source_impedance_depth_array = np.array([source_depth_list_mm[::-1]]).T * 1e-3

        lm.distance = np.sum(lm.depth_array)

        return lm

    @staticmethod
    def plot_layer_compare(layer_models: List, labels: List) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot a Tissue ID vs. Distance plot to compare different layer models
        
        :param labels:
        :param layer_models:
        :return:
        """

        (hf, ha) = plt.subplots()

        # plot the resulting layer models
        for (lm, label) in zip(layer_models, labels):
            ha.plot(np.hstack((np.flip(-np.cumsum(np.flip(lm.source_impedance_depth_array, axis=0)), axis=0),
                              np.cumsum(lm.depth_array))),
                    np.hstack((np.flip(lm.source_impedance_tissue_array, axis=0), lm.tissue_array)), label=label)

            ha.set_xlabel('Total Distance, TX placed at d=0')
            ha.set_ylabel('Tissue ID')
            ha.legend()

        return hf, ha


class LayerModel(LayerModelBase):

    def __init__(self, vm: VoxelModel = VoxelModel(),
                 startpoint: Coordinate = Coordinate([0, 0, 0]),
                 endpoint: Coordinate = Coordinate([1, 1, 1]),
                 tissue_properties: DielectricProperties = TissueProperties(), lm_closest: DirectLinkLayerModel = None,
                 debug: bool = False, on_body_pl_antenna_height: str = '5mm',
                 model_type: str = 'trunk'):

        """
        An extended version of the layer model taking also an indirect path into consideration.

        :param vm:
        :param startpoint:
        :param endpoint:
        :param tissue_properties:
        :param lm_closest: A DirectLinkLayerModel object containing the layer model for the path from startpoint to closest
                           endpoint. This should speed up some computations as this is the same for each startpoint.
        :param debug: Activate some debug plots showing the geometry (does only work in start and endpoint have same
                      z-coordinate.
        :param on_body_pl_antenna_height: height of the on-body antenna over the body surface. May be '0mm' or '5mm' as
                                          specified in the on-body path loss model.
        :param model_type: determines which model_type inside the VoxelModel to use for computations.
        """
        self.on_body_pl_antenna = on_body_pl_antenna_height

        super().__init__(vm, startpoint, endpoint, model_type)

        # compute the direct link layer model
        self.lm_direct = DirectLinkLayerModel(vm, startpoint, endpoint, tissue_properties, model_type)
        # The distance of the total layer model is nonetheless the direct distance:
        self.distance = self.lm_direct.distance

        # find the closest receive antenna location on the abdominal surface and compute the layermodel
        endpoint_list = [Coordinate(s['centroid']) for s in vm.models['trunk'].surface_3d]
        closest_endpoint_idx = np.argmin(cdist(np.array(startpoint).reshape(1, 3),
                                               np.array(endpoint_list),
                                               'euclidean'))
        closest_endpoint = endpoint_list[closest_endpoint_idx]

        # if the endpoint is already the closest we can skip the rest
        if closest_endpoint == endpoint:
            self.extension_used = False
            self.onbody_dist = 0
            self.closest_endpoint = endpoint
            self.closest_midpoint = None
            self.lm_indirect = None
            self.lm_closest = None
        else:
            # layer model to closest endpoint
            if lm_closest is None:
                self.lm_closest = DirectLinkLayerModel(vm, startpoint, closest_endpoint, tissue_properties, model_type)
            else:
                self.lm_closest = lm_closest
                self.closest_endpoint = lm_closest.endpoint

            # get the distance between the closest endpoint on the surface and the actual endpoint:
            # Construct a circle out of endpoint - closest_endpoint - A
            # where A is the point on the body surface that is closest to the midpoint of the
            # line from endpoint to closest_endpoint
            midpoint = endpoint + (closest_endpoint - endpoint) * 1 / 2
            # closest point on surface to midpoint
            closest_midpoint = endpoint_list[np.argmin(cdist(np.array(midpoint).reshape(1, 3),
                                                             np.array(endpoint_list),
                                                             'euclidean'))]

            # it can happen that the closest_midpoint is the same as the closest_endpoint -> the endpoint and
            # closest_endpoint are close together anyway -> no circle approximation needed
            if closest_endpoint == closest_midpoint or closest_midpoint == endpoint:
                air_dist = np.linalg.norm(closest_endpoint - endpoint) / 1e3
                r, beta = None, None
            else:
                # get the radius and the angle of the circle segment
                # of the outer circle of the triangle endpoint - closest_midpoint - closest_endpoint
                r, beta = self.get_radius_circle(closest_endpoint, closest_midpoint, endpoint)
                # opening angle of the circular arc
                delta = 2*np.pi - 2*beta
                # distance along the circle described by endpoint - closest_midpoint - closest_endpoint
                air_dist = r * delta / 1e3

            if debug:
                hf, ha = vm.plot_slice(slice_number=vm.models['trunk'].coordinate_to_index(startpoint, vm.scaling)[2])
                ha.plot(startpoint.y, startpoint.x, marker='X', label='start')
                endpoint_slice = [e for e in endpoint_list if e.z == closest_endpoint.z]
                for e in endpoint_slice:
                    ha.plot(e.y, e.x, marker='o', fillstyle='none')
                ha.plot(endpoint.y, endpoint.x, marker='X', label='end')
                ha.plot(closest_midpoint.y, closest_midpoint.x, marker='s', label='closest_midpoint')
                ha.plot(closest_endpoint.y, closest_endpoint.x, marker='o', label='closest_endpoint')
                ha.legend()
                if r is not None:
                    print(f'radius = {r}, beta = {beta * 180 / np.pi}, air_dist = {air_dist*1e3} mm')
                else:
                    print(f'air_dist = {air_dist * 1e3} mm')

            self.closest_endpoint = closest_endpoint
            self.closest_midpoint = closest_midpoint
            self.onbody_dist = air_dist

            # Build the layer model of the indirect path: start -> closest_endpoint -> endpoint
            self.lm_indirect = DirectLinkLayerModel(layermodel=self.lm_closest)
            self.lm_indirect.depth_array = np.append(self.lm_indirect.depth_array, air_dist)
            self.lm_indirect.tissue_array = np.append(self.lm_indirect.tissue_array,
                                                      tissue_properties.get_id_for_name('Air'))
            self.lm_indirect.endpoint = endpoint
            self.lm_indirect.distance = np.sum(self.lm_indirect.depth_array)

    def onbody_loss(self, distance: float):
        """
        The UWB on-body propagation loss along the torso from
            [YS10] K. Y. Yazdandoost and K. Sayrafian-Pour, Channel Model for Body Area Network.
            [Online] Available: https://mentor.ieee.org/802.15/dcn/08/15-08-0780

        Return value is in linear scale and for use with magnitudes (not powers!)
        :param distance:
        :return:
        """
        if self.on_body_pl_antenna == '5mm':
            pl0 = 44.6
        elif self.on_body_pl_antenna == '0mm':
            pl0 = 56.5
        else:
            raise ValueError("'on_body_pl_antenna' has to be '5mm' or '0mm'.")

        return 10 ** ((pl0 + 10 * 3.1 * np.log10(distance / 0.1)) / 20)

    def transfer_function(self,
                          f_start: float = 1e9,
                          f_end: float = 10e9,
                          n_samples: int = 2048,
                          radiation_loss: str = None,
                          frequency_vector: Union[str, np.ndarray] = 'natural',
                          endpoint: bool = False, debug: bool = False, tf_component: str = 'sum',
                          **kwargs) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the transfer function of the LayerModel at n equidistant frequency points from f_start to f_end.

        :param float f_start:      lower frequency
        :param float f_end:        higher frequency
        :param int n_samples:   number of evaluation points
        :param radiation_loss:  Select how to calculate the additional radiation loss that is added to the transfer
                                functions. Possible values are: 'air' (assume speed of light in free space) or
                                'effective_tissue' (compute phase velocity from effective propagation delay in the
                                tissue layers).
        :param frequency_vector:    May be a numpy array containing all the frequency samples or a string. Then
                                    the value determines how the frequencies between f_start and f_end are computed.
                                    if frequency_vector == 'natural'
                                        ==> f = linspace(f_start, f_end, n_samples, endpoint = endpoint)
                                    if frequency_vector == 'fft'
                                        ==> bw = (f_end - f_start)
                                            f = np.fft.fftfreq(n_samples, 1/bw) + fc

        :param endpoint: include f_end in the calculation (default False)
        :param debug: Activate some debug plots showing the geometry (does only work in start and endpoint have same
                      z-coordinate.
        :param tf_component: Specify the transfer function to return: either 'sum' (default), 'direct', or 'indirect'
        :return: A tuple (tf, f).
        """
        # Transfer function of direct link including radiation loss
        tf_direct, f = self.lm_direct.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples,
                                                        radiation_loss=radiation_loss,
                                                        frequency_vector=frequency_vector, endpoint=endpoint)

        # transfer function of indirect link over closest endpoint without radiation loss
        # -> add the radiation loss below
        if self.lm_indirect is not None:
            tf_indirect, f = self.lm_indirect.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples,
                                                                radiation_loss='none',
                                                                frequency_vector=frequency_vector,
                                                                endpoint=endpoint)

            rl = self.lm_closest.radiation_loss(type=radiation_loss, f_start=f_start, f_end=f_end,
                                                n_samples=n_samples, frequency_vector=frequency_vector)

            tf_indirect = tf_indirect / rl
            if self.onbody_dist > 0:
                tf_indirect /= self.onbody_loss(self.onbody_dist)
        else:
            tf_indirect = np.zeros(tf_direct.shape)

        # total transfer function is sum of both
        transfer_function = tf_direct + tf_indirect

        if debug:
            from plots import ComplexPlot
            hf, ha = plt.subplots(2, 2)
            p = ComplexPlot(type='mag_delay', db=20, hf=hf, ha=ha[:, 0])
            p.plot(f, tf_direct, label='tf_direct', linestyle='--')
            p.plot(f, tf_indirect, label='tf_closest2end', linestyle='--')
            p.plot(f, transfer_function, label='transfer_function')
            p = ComplexPlot(type='real_imag', hf=hf, ha=ha[:, 1])
            f_start_idx = np.argmin(np.abs(f - 3.1e9))
            f_end_idx = np.argmin(np.abs(f - 4.8e9))
            p.plot(*self.lm.impulse_response(f_start=3.1e9, f_end=4.8e9,
                                             precalculated_tf=(tf_direct[f_start_idx:f_end_idx],
                                                               f[f_start_idx:f_end_idx]))[::-1],
                   label='tf_direct', linestyle='--')
            p.plot(*self.lm.impulse_response(f_start=3.1e9, f_end=4.8e9,
                                             precalculated_tf=(tf_indirect[f_start_idx:f_end_idx],
                                                               f[f_start_idx:f_end_idx]))[::-1],
                   label='tf_closest2end', linestyle='--')
            p.plot(*self.lm.impulse_response(f_start=3.1e9, f_end=4.8e9,
                                             precalculated_tf=(transfer_function[f_start_idx:f_end_idx],
                                                               f[f_start_idx:f_end_idx]))[::-1],
                   label='transfer_function', linestyle='-')

        if tf_component == 'sum':
            return transfer_function, f
        elif tf_component == 'direct':
            return tf_direct, f
        elif tf_component == 'indirect':
            return tf_indirect, f
        else:
            raise ValueError("Possible values of tf_return are 'sum' (default), 'direct', or 'indirect'")

    def propagation_delay(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                          frequency_vector: Union[str, np.ndarray] = 'natural', endpoint: bool = True) -> Dict:
        """
        Compute the propagation delay through the direct and the indirect layer model for a given frequency range.
        All parameters are handed to self.frequency_vector().

        :return:
        """
        tau_ges = {}
        for lm_type in ['direct', 'indirect']:
            lm = getattr(self, 'lm_' + lm_type)
            tau_ges[lm_type] = lm.propagation_delay(f_start, f_end, n_samples, frequency_vector, endpoint)

        return tau_ges

    def phase_velocity(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                       frequency_vector: Union[str, np.ndarray] = 'natural', endpoint: bool = True) -> Dict:
        """
        Compute the phase velocity through the direct and the indirect layer model for a given frequency range.
        All parameters are handed to self.frequency_vector().

        :return:
        """
        tau_ges = self.propagation_delay(f_start, f_end, n_samples, frequency_vector, endpoint)

        phase_velocity = {}
        for lm_type, tau in tau_ges.items():
            lm = getattr(self, 'lm_' + lm_type)
            phase_velocity[lm_type] = lm.distance / tau

        return phase_velocity

    @staticmethod
    def get_radius_circle(a: Coordinate, b: Coordinate, c: Coordinate) -> Tuple[float, float]:
        """
        Get the radius of a circle from 3 points in 3D using the law of sines in a triangle.

        :param a:
        :param b:
        :param c:
        :return:
        """
        norm = np.linalg.norm
        ba = a - b
        ac = c - a
        bc = c - b
        # angle at point b
        # here the angle is always > 90° and < 180° ==>
        temp = np.around(np.dot(ba, bc) / (norm(ba) * norm(bc)), 8)  # round to 6 decimal places to avoid errors
        if not (-1 <= temp <= 1):
            raise ValueError("Argument of the arccos has to be in [-1, 1]. ")

        beta = np.arccos(temp)
        radius = norm(ac) / (2 * np.sin(beta))
        return radius, beta
