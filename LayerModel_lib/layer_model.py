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

from typing import Tuple, Optional, Union, Dict, List

from LayerModel_lib.voxelmodel import VoxelModel
from LayerModel_lib.dielectric import DielectricProperties
from LayerModel_lib.tissue_properties import TissueProperties
from LayerModel_lib.coordinate import Coordinate


class LayerModel:
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

    def __init__(self, voxel_model: VoxelModel=VoxelModel(),
                 startpoint: Coordinate=Coordinate([0, 0, 0]), endpoint: Coordinate=Coordinate([1, 1, 1]),
                 tissue_properties: DielectricProperties=TissueProperties(),
                 model_type: str = 'trunk'):
        """
        Generate the layer model for voxel_model

        :param TissueProperties tissue_properties:  Object containing the dielectric tissue properties
        :param VoxelModel voxel_model:              A VoxelModel object that is used for the calculation of the layers
        :param numpy.ndarray startpoint:            Startpoint coordinate (transmitter location) inside the VoxelModel
        :param numpy.ndarray endpoint:              Endpoint coordinate (receiver location) outside the VoxelModel
        :param model_type:   the type of the model to use, e.g. 'complete' or 'trunk'
        """
        # The tissue property object is used to calculate the dielectric properties
        self.tissue_properties = tissue_properties  # TissueProperties()
        # fix the startpoint
        self.startpoint = startpoint

        self.tissue_array = np.zeros(shape=(0,))
        self.depth_array = np.zeros(shape=(0,))
        self.endpoint = endpoint

        self.source_tissue = 0
        # if an empy layer model is created, assume a layer of 1000mm air as source medium
        self.source_impedance_tissue_array = np.array([0])
        self.source_impedance_depth_array = np.array([[1000]]) * 1e-3
        self.source_impedance_load_tissue = np.array([0])
        # the distance between transmitter and receiver
        self.distance = 0

        self.vm = voxel_model
        # check if the voxel_model is empty
        # only get the tissue layers from the model if it is not empty
        # otherwise the tissue layers need be set individually by hand
        if voxel_model.name is not 'empty':
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
            logging.warning("Distance for source impedance calculation was reduced fromm 100 mm to %d mm" % distance)

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
        print("========= LayerModel info =========\n\n"
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
                          f_start: float = 0.0,
                          f_end: float = 10e9,
                          n_samples: int = 2048,
                          field_type: str = 'S21',
                          direction: str = 'start->end',
                          radiation_loss: str = None,
                          free_space_loss_distance: float = None)\
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Calculate the transfer function for either the electric or magnetic field at n equidistant frequency points
        from f_start to f_end.
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
        :return:
        """
        # generate the frequency values
        f_theoretic = np.linspace(f_start, f_end, n_samples)
        f_theoretic.shape = (-1, 1)

        depth_array = np.array([self.depth_array])
        depth_array.shape = (1, -1)

        # if f_start is zero this needs to be replaced by 100 Hz for calculation
        if f_start == 0:
            f = np.vstack([100, f_theoretic[1:]])
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
            transfer_function, f_theoretic = self.S21(f_start=f_start, f_end=f_end, n_samples=n_samples)

            c0 = 299792458  # m/s in free space!
            c = None
            if radiation_loss == 'air':
                c = c0
                d = self.distance
            elif radiation_loss == 'avg_tau':
                # determine speed of light by calculating average speed of light from group delay
                tau_g = np.mean(- np.gradient(np.unwrap(np.angle(transfer_function.flat)), f_theoretic.flat)
                                / (2 * np.pi))
                c = self.distance / tau_g
                d = self.distance

            if radiation_loss is None and free_space_loss_distance is not None:
                c = c0
                d = free_space_loss_distance

            # c is only not None if some additional losses need to be added.
            if c is not None:
                wavelength = c / f
                transfer_function = (wavelength / (4 * np.pi * d)) * transfer_function
        else:
            raise ValueError("'field_type' can only be 'S21', 'E' or 'H'")

        return transfer_function, f_theoretic

    def S21(self, f_start: float=0.0, f_end: float=10e9, n_samples: int=2048)\
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate S21 for this layer model at n equidistant frequency points from f_start to f_end

        :param float f_start:      lower frequency
        :param float f_end:        higher frequency
        :param int n_samples:   number of evaluation points

        :return:  numpy.ndarray    S21 for this layer model and the corresponding frequency vector (S21, f)
        """
        # generate the frequency values
        f_theoretic = np.linspace(f_start, f_end, n_samples)
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

        return S21, f_theoretic

    def impulse_response(self, f_sample: float = 20e9,
                         n_samples: int = 2049,
                         normalize_energy: bool = False,
                         truncation_percentage: float = 1,
                         normalize_delay: str = 'none',
                         threshold_dB: float = 30,
                         direction: str = 'start->end',
                         equivalent_baseband: Dict = None,
                         precalculated_S21_f: Tuple = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the impulse response for this LayerModel with the given sampling rate. All kwargs are passed into
        self.transfer_function().

        :param float f_sample:                The sampling rate of the impulse response.
        :param int n_samples:                 The number of samples of the underlying transfer function/ifft calculation
        :param bool normalize_energy:         Boolean to normalize the energy of the impulse response to 1
        :param float truncation_percentage:   Value in [0,1] to determine after which percentage of the energy the
                                              output gets truncated.
        :param str normalize_delay:           Determines if the delay of the impulse response is kept ('none') or if the
                                              output is truncated such that the rising edge ('edge') or the
                                              peak ('peak') is at t=0
        :param str direction:   sets the direction of the impulse response, either from startpoint to endpoint or
                            vice versa. Can either be 'start->end' or 'end->start'
        :param float threshold_dB: threshold for determining the start of the impulse response if normalize_delay is
                                    set to 'edge'.
        :param equivalent_baseband: Used to convert the impulse response to equivalent baseband represantation.
                                    If this is not None it is assumed to be a dictionary with the following keys
                                    {'fc': carrier frequency of downconversion,
                                     'B': bandwidth for the equivalent baseband signal, centered at fc}.
                                    f_sample has no meaning in this case, as the new sampling rate is 2*B.
        :param Tuple precalculated_S21_f: A tuple containing (S21, f) calculated already with self.S21()

        :return: impulse response, t:  The impulse response and corresponding time vector t
        :rtype: Tuple[numpy.ndarray, numpy.ndarray]
        """

        if equivalent_baseband is None:
            f_start = 0
            f_end = f_sample/2
        else:
            # define variables for downconversion
            bandwidth = equivalent_baseband['B']
            carrier_frequency = equivalent_baseband['fc']
            f_start = carrier_frequency - bandwidth/2
            f_end = carrier_frequency + bandwidth/2
            f_sample = bandwidth
            f_BB = np.linspace(-bandwidth/2, bandwidth/2, n_samples)  # frequencies in eq. baseband

        if precalculated_S21_f is None:
            if direction != 'start->end':
                raise ValueError("For field_type=S21 only the direction 'start->end' is supported.")

            transfer_function, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (transfer_function, f) = precalculated_S21_f

        # make sure that transfer_function is a column vector
        transfer_function.shape = (-1, 1)

        if equivalent_baseband is None:
            # Take the real IFFT irfft, as we want the output to be real and assume the spectrum to be Hermitian
            # as long as we compute the impulse response in the passband
            impulse_response = np.fft.irfft(transfer_function, axis=0)
        else:
            # For equivalent baseband we need to take the IFFT, but first reorder the transfer function, such that the
            # zero frequency component is in tf_BB[0].
            tf_BB = np.fft.ifftshift(transfer_function)
            impulse_response = np.fft.ifft(tf_BB, axis=0)

        # Total energy of impulse response
        energy = np.sum(impulse_response ** 2, axis=0)

        # normalize energy of impulse response to 1
        if normalize_energy:
            impulse_response = impulse_response / np.sqrt(energy)
            energy = 1

        # If the impulse response is supposed to be truncated after some percentage of energy
        # truncation_percentage. If truncation_percentage >= 1 no truncation is done
        if truncation_percentage < 1:
            k_max = 0
            for k in range(0, impulse_response.shape[0]):
                energy_k = np.sum(impulse_response[0:k, :] ** 2, axis=0)
                if energy_k > energy * truncation_percentage:
                    k_max = k
                    break

            if k_max != 0:
                impulse_response = impulse_response[0:k_max, :]

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
        t = np.linspace(0, impulse_response.size / f_sample, impulse_response.size)
        t.shape = (-1, 1)

        return impulse_response, t

    def path_loss(self, f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                  precalculated_S21_f: Tuple = None, **kwargs) -> float:
        """
        Calculate the path loss in the frequency band between f_start and f_end using n_samples.
        It is assumed for this function that the transmit power spectral density is uniformly distributed
        between f_start and f_end. All **kwargs will be passed to self.transfer_function().

        :param float f_start:     lower frequency
        :param float f_end:       higher frequency
        :param int n_samples:   number of samples to use to evaluate the path loss
        :param Tuple precalculated_S21_f: A tuple containing (S21, f) calculated already with self.S21()
        :return: float: the path loss
        """

        bandwidth = f_end - f_start

        if precalculated_S21_f is None:
            S21, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (S21, f) = precalculated_S21_f

        path_gain = np.trapz(abs(S21) ** 2, x=f, axis=0) / bandwidth
        path_loss = 1 / path_gain

        return path_loss

    def channel_capacity(self, noise_power_density: float, transmit_power: float,
                         f_start: float = 3.1e9, f_end: float = 4.8e9, n_samples: int = 1024,
                         precalculated_S21_f: Tuple = None, **kwargs) -> float:
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
        :param Tuple precalculated_S21_f: A tuple containing (S21, f) calculated already with self.S21()
        :return:
        """

        bandwidth = f_end - f_start

        if precalculated_S21_f is None:
            S21, f = self.transfer_function(f_start=f_start, f_end=f_end, n_samples=n_samples, **kwargs)
        else:
            (S21, f) = precalculated_S21_f

        path_gain = np.abs(S21) ** 2

        # the energy per symbol is the magnitude of the power spectral density
        energy_per_symbol = transmit_power / (2 * bandwidth)
        transmit_power_density = energy_per_symbol * np.ones(S21.shape)

        capacity = np.trapz(np.log2(1 + 2 * (transmit_power_density * path_gain) / noise_power_density), x=f, axis=0)

        return capacity

    @staticmethod
    def create_from_dict(layers: list, tissue_properties: DielectricProperties = TissueProperties()) \
            -> 'LayerModel':
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
                        The example dict above results in the following LayerModel:
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
        lm = LayerModel(tissue_properties=tissue_properties)

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
