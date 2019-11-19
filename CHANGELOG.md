# Changelog
All notable changes to the **LayerModel_lib** project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

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
 