# Changelog
All notable changes to the **LayerModel_lib** project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

##  [2.4.0] 
### Changed
- add all dielectric properties to install using MANIFEST.in
- update README for published version of thesis
- some cosmetic changes for plotting the abdominal surfaces of the phantoms

##  [2.3.0] - 2023-01-19
### Added
- Polynomial Channel Model and its parameters for all phantoms and TX/RX locations

##  [2.2.0] - 2020-09-18
### Added
- Function `LayerModelBase.frequency_vector()` to simplify the creation of frequency vectors for the computation.
- Function `DirectLinkLayerModel.radiation_loss()` to compute the radiation loss with either phase velocity in air or determined from the tissue properties. This works now properly and computes the (average) frequency dependent phase velocity through all tissue layers directly from the propagation constant gamma. NOT by using the group delay anymore -> that is not physicall.
- `force_overwrite` argument to `SimulationScenario.save()`.
- Parameter `truncation` in `LayerModelBase.impulse_response()` to select the possible truncation after a 
certain threshold of the impulse response energy is crossed.
 
### Fixed
- `LayerModelBase.path_loss()` bandwidth calculation was wrong
- A bug that only 8 attributes could be saved. This was caused by track_orders=True. 
see https://github.com/h5py/h5py/issues/1385 for details.

 

##  [2.1.0] - 2020-09-09
### Added:
- `SimulationScenario`s are now saved as HDF5 files with compression. The old
pickled format is still supported.
- Function `SimulationScenario.delete_result()` to delete results from the scenario.

##  [2.0.0] - 2020-09-04
### Major Changes:
- renamed `LayerModel` to `DirectLinkLayerModel` and
- renamed `LayerModelExtended` to `LayerModel` as this now the final layer model

### Added
- new parameter `on_body_pl_antenna_height` for `LayerModel` for calculation of the on-body path loss
- added `distance` attribute to `LayerModel`
### Fixed
- computation of opening angle between direct link and indirect link in `LayerModel.get_radius_circle()` improved, by rounding the value that is put into the arccos() function.

##  [1.3.0] - 2020-06-24
### Added
- Import script for Alvar including clustering of abdominal endpoints
- clean-up function for 3D surface to remove artifacts that may be generated
- New class `LayerModelExtended` that takes the direct as well as the indirect path from TX to RX into account.

### Fixed
- All skin of AustinMan and AustinWoman is now set to SkinWet

##  [1.2.2] - 2020-02-18
### Fixed
- Missing **kwargs argument for `LayerModel.impulse_response()` to include the effect of radiation loss  

##  [1.2.1] - 2019-11-19
### Added
- New static methods to create and load a `SimulationScenario`
- Add various parameters to `LayerModel.show_3d_model()` that influence how the model is plotted
- Add **kwargs to `LayerModel.path_loss()` and `LayerModel.capacity()` to pass arguments to transfer function.
Hence, allow to include radiation loss into path loss and capacity calculations.

### Fixed
- Some minor bugfixes and cleaning up of some parts of the code

##  [1.1.0] - 2019-10-18
### Added
- `LayerModel.transfer_function()` has a new paramter `radiation_loss` to determine how additional radiation loss of
the form $`RL = \left(\frac{4\pi fd}{\tilde{c}}\right)`$. The new parameter `radiation_loss` determines how
$`\tilde{c}` is computed.

### Fixed
- Missing `SimulationScenario` raises a `FileNotFoundError`
- Minor bugfixes

##  [1.0.0] - 2019-04-25
### Added
- Function LayerModel.create_from_dict() for easier creating layer models with arbitrary tissue layers.

### Fixed
- A lot of minor bugfixes

## [0.8.0] - 2019-02-28
### Added
- Physiological properties to `VoxelModel`
- Plot functionality in `LayerModel.plot()` and `LayerModel.plot_layer_compare()`
### Fixed
- Multiple small bugfixes in all files.
