# This file is part of LayerModel_lib
#
#     A tool to compute the transmission behaviour of plane electromagnetic waves
#     through human tissue.
#
# Copyright (C) 2018 Jan-Christoph Brumm
#
# Licensed under MIT license.
#
"""
This example shows how the complex permittivity of the different tissues is computed.
"""

import numpy as np
import matplotlib.pyplot as plt

from LayerModel_lib import TissueProperties

tp = TissueProperties()

# print all available tissues and their relative permittivity at 4 GHz
for (i, t) in enumerate(tp.tissue_names):
    print('%d %s: %.4f' % (i, t, np.real(tp.complex_permittivity(i, 4e9) / tp.epsilon0)))

# get the id of muscle
muscle_id = np.array(tp.get_id_for_name(['Muscle', 'Fat']))

# calculate permittivity for vector f
f = np.linspace(1e6, 10e9, 500)
eps = tp.complex_permittivity(muscle_id, f) / tp.epsilon0

fig, ax = plt.subplots(nrows=2)
ax[0].semilogy(f, np.real(eps))
ax[0].set_title("Real Part")
ax[1].semilogy(f, -np.imag(eps))
ax[1].set_title("negative Imag. Part")
plt.show()

