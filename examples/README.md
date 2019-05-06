# Examples how to use the LayerModel_lib
This folder contains several different examples on how to use the different modules of 
the LayerModel_lib package. 

##  List of all Examples
| Filename | Description |
|----|---|
|[draw_slice_of_voxel_model.py](draw_slice_of_voxel_model.py)| Draw one transverse slice of a voxel model |
|[generate_CST_model.py](generate_CST_model.py)| Export an arbitrary VoxelModel to a *CST Microwave Studio* compatible binary file |
|[impulse_response_from_voxelmodel.py](impulse_response_from_voxelmodel.py)|Compute the impulse response between two specific points |
|[permittivity.py](permittivity.py)| Compute the complex permittivity for different tissues and plot it | 
|[random_transfer_function_from_voxelmodel.py](random_transfer_function_from_voxelmodel.py) | Randomly draw some start and endpoint inside a VoxelModel and calculate the transfer function of this setup |
|[simulation_scenario_usage.py](simulation_scenario_usage.py)| How to use the class `SimulationScenario` for storing simulation results.|
|[transfer_function_from_custom_layer_setup.py](transfer_function_from_custom_layer_setup.py)|Define a custom layer setup of different tissues and calculate the transfer function|
|[transfer_function_from_voxelmodel.py](transfer_function_from_voxelmodel.py)|Compute the transfer function between two specific points |
|[transfer_function_with_constant_permittivity.py](transfer_function_with_constant_permittivity.py)|Define a custom layer setup with dielectrics that have *frequency independent* permittivity |
|[transfer_function_with_permittivity_and_conductivity.py](transfer_function_with_permittivity_and_conductivity.py)|Define a custom layer setup with dielectrics that have permittivity $`\varepsilon = \varepsilon_0\cdot \varepsilon_r' + j\cdot \frac{\sigma}{2\pi f}`$|
|[visualize_layer_model.py](visualize_layer_model.py)| Show the two different possibilities on how to visualize the layer models. |