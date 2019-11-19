[![DOI](https://zenodo.org/badge/191758079.svg)](https://zenodo.org/badge/latestdoi/191758079)


# LayerModel_lib - A Python Toolkit to Compute the Transmission Behaviour of Plane Electromagnetic Waves Through Human Tissue

## Introduction and Motivation

This repository provides a library to calculate transfer functions, path loss, channel capacity and various 
other properties of in-body to on-body ultra wideband commincation using the layer modeling approach as introduced 
in [BB17]. Instead of simulating the wave propagation inside the human body using numerical methods, the transmission 
form transmitter TX to receiver RX is simplified by a plane wave travelling through a multi-layered dielectric.
The layers of this multi-layered dielectric can be determined from arbitrary voxel models.  

It was shown in [BB17b], [BKB19], and [BB19] that the results from this layer modeling approach fit well to the results that 
have been published in the literature so far for similar transmission setups. For more details on this modeling
technique refer to [BB17b] or [TT+12]. The actual results in [BB17a], [BB17b], [BB17c] and [BKB19] were computed using MATLAB. 
`LayerModel_lib` is a Python implementation of the same functionality. 

The results from [BB19] were simulated using this library and the resulting data 
can be found in int/in-body/ismict2019>.

## Citation
This code is distributed under [MIT license](LICENSE). When using this code or parts of it for publications or research
please cite this repository as:

[Bru19] J.-C. Brumm, "LayerModel_lib. A Python toolkit to compute the transmission behaviour of plane 
electromagnetic waves through human tissue." DOI: 10.5281/zenodo.3507610 

## Installation and Requirements
1. Download and install at least Python 3.6
2. The following Python packages are required:
   1. `scipy`
   2. `numpy`
   3. `texttable`
   4. `progressbar2`
   5. `matplotlib`
   6. `opencv-python`
   7. `scikit-learn`
   8. `networkx`
   
   When using Anaconda the supplied `environment.yml` can be used to create a new environment with all dependencies. 
2. Clone the git repository
3. Install the `LayerModel_lib` package using
```commandline
python setup.py install
```

## Voxel Models
Due to license restrictions no voxel models are included in this package. In [phantom_import/Readme.md](phantom_import/Readme.md) 
a list of all supported voxel models can be found together with a detailed explanation how to import the voxel models. 
For all of the models mentioned therein an import script is available in the folder `phantom_import`.

## Example
A short example on how to calculate the transfer function between a transmitter placed in the
gastrointestinal tract and a receiver on the body surface is given in the following:

```python
import os
from LayerModel_lib import VoxelModel, LayerModel

# set the working directory where the imported voxel models can be found 
VoxelModel.working_directory = os.path.join('..', 'Phantoms')
# Load a virtual human model
vm = VoxelModel('AustinWoman_v2.5_2x2x2')

# generate 10 random endpoints (receiver locations) on the abdominal surface
e = vm.get_random_endpoints('trunk', 10)

# get 10 random startpoints (transmitter locations) in the gastrointestinal tract
s = vm.get_random_startpoints('trunk', 'GIcontents', 10)

# calculate the layer model between two of these points
lm = LayerModel(voxel_model=vm, startpoint=s[1], endpoint=e[1])

# compute the transfer function 
transfer_function, f = lm.transfer_function()
```

More examples can be found in the [examples](examples/README.md) folder.

## References 

[BB19] J.-C. Brumm, and G. Bauch, “Influence of Physiological Properties on the
Channel Capacity for Ultra Wideband In-Body Communication,” in *13th International Symposium on Medical 
Information and Communication Technology (ISMICT'2019)*. Oslo, 2019.

[BKB19] J.-C. Brumm, J. Kohagen, and G. Bauch, “Improving Ultra Wideband In-
Body Communication Using Space Diversity,” in *12th International ITG
Conference on Systems, Communications and Coding 2019 (SCC’2019)*. Rostock, 2019.

[BB17c] J.-C. Brumm and G. Bauch, “On the Shadowing Distribution for Ultra Wideband 
In-Body Communication Path Loss Modeling,” in *IEEE AP-S Symposium on Antennas and Propagation 
and USNC-URSI Radio Science Meeting*. San Diego, USA, July 2017.

[BB17b] J.-C. Brumm and G. Bauch, “On the Placement of On-Body Antennas for
Ultra Wideband Capsule Endoscopy,” *IEEE Access*, vol. 5, pp. 10141–10149, 2017. 
http://dx.doi.org/10.1109/ACCESS.2017.2706300

[BB17a] J.-C. Brumm and G. Bauch, “Channel Capacity and Optimum Transmission Bandwidth 
of In-Body Ultra Wideband Communication Links,” in *11th International ITG Conference on 
Systems, Communications and Coding 2017 (SCC’2017)*. Hamburg, Germany, 2017.

[TT+12] P. Theilmann, M. A. Tassoudji, E. H. Teague, D. F. Kimball, and P. M. Asbeck, 
“Computationally Efficient Model for UWB Signal Attenuation Due to Propagation in Tissue 
for Biomedical Implants,” *Progress In Electromagnetics Research B*, vol. 38, pp. 1–22, 2012.
