# Voxel Model Phantoms

There are 2 main sources for the models in this project.

1. [Helmholtz Zentrum München Virtual Human Database](https://www.helmholtz-muenchen.de/irm/service/virtual-human-download-portal/virtual-human-database/index.html)
2. The University of Texas: [AustinMan](http://web2.corral.tacc.utexas.edu/AustinManEMVoxels/AustinMan/index.html) and [AustinWoman](http://web.corral.tacc.utexas.edu/AustinManEMVoxels/AustinWoman/index.html)

## List of all available Phantoms
The following table gives an overview over the properties of the different models. The upper 12 are available from the Helmholtz Zentrum München and the last two from the University of Texas.


| Name |Import Available |  Sex | Age | Size (cm) | Weight (kg) | Resolution |
|----- |---- | ---- |---- |---------- |-------------| -----------|
|Baby  | :heavy\_multiplication\_x: | W    | 8w  | 57 | 4.2 | |
|Child | :heavy\_multiplication\_x: | W    | 7   | 115| 21.7 | |
|Jo    | :heavy\_multiplication\_x: | W    | 8   | 130|34   | |
|Helga | :heavy\_check\_mark: | W    |26   | 170|81 | $`(0.98\cdot 0.98\cdot 10)mm^3 `$ |
|Irene | :heavy\_check\_mark: | W    | 32  | 163|51| $` (1.875\cdot 1.875\cdot 5)mm^3`$ |
|VisibleHuman | :heavy\_check\_mark: | M | 38 |180 | 103 | $`(0.91 \cdot 0.94 \cdot 5) mm^3`$ |
|Golem (only Small Intestine labeled, w/o the contents) | :heavy\_check\_mark: | M | 38 |176 | 69 | $`(2.08 \cdot 2.08 \cdot 8) mm^3`$ |
|Donna | :heavy\_check\_mark: | W | 40 | 170 | 70 | $`(1.875 \cdot 1.875 \cdot 10) mm^3`$ |
|Frank [^1] | :heavy\_check\_mark: | M | 48 | n.A. | n.A.| $` (0.742188\cdot 0.742188\cdot 5)mm^3`$ 3D Model not complete. |
|Katja [^2] | :heavy\_check\_mark: | W | n.A. | n.A. | n.A. | $` (1.775\cdot 1.775\cdot 4.84)mm^3`$ |
|ADAM | :heavy\_multiplication\_x: | M |  n.A. | n.A. | n.A. | $`(1.6 \cdot 1.6\cdot 2)mm^3`$ |
|EVA | :heavy\_multiplication\_x: | W |  n.A. | n.A. | n.A. | |
|AustinMan | :heavy\_check\_mark: :heavy\_check\_mark: :heavy\_multiplication\_x: :heavy\_multiplication\_x: | M | 38 | 180 | 106.2 | $`(1\cdot 1\cdot 1)mm^3, (2\cdot 2\cdot 2)mm^3, (4\cdot 4\cdot 4)mm^3, (8\cdot 8\cdot 8)mm^3`$ |
|AustinWoman | :heavy\_check\_mark: :heavy\_check\_mark: :heavy\_multiplication\_x: :heavy\_multiplication\_x: | W | 59 | 173 | 84.8 | $`(1\cdot 1\cdot 1)mm^3, (2\cdot 2\cdot 2)mm^3, (4\cdot 4\cdot 4)mm^3, (8\cdot 8\cdot 8)mm^3`$ |

[^1]: The model Frank is not complete in the transverse plane i.e., parts of the abdomen are cut off. 
[^2]: Katja is pregnant and for some reason the fat tissue is labeled as body fluid. 

## Import and Usage of Voxel Model Phantoms

For all Phantoms with a :heavy\_check\_mark: in the column "Import Available" a script ImportXY.py can be found in this
 folder. Only the path to the actual model needs to be replaced by the respective location of the (unzipped) model, e.g. from the Virtual Human Database or the AustinMan/Woman website.
 
### Import 
1. Download the desired voxel model and unzip it, e.g. Donna. 
2. Change the path in the corresponding import script.
3. Run the import script, e.g. `python ImportDonna.py`. This may take some time, especially the generation of the 
3D surface. 
4. Copy the resulting file, e.g. `Donna.VoxelModel`, to a known folder

After importing all the models, the possible receiver locations on the abdominal surface need to be added. This is done
by running `add_receiver_locations_clustering.py`. This script will automatically determine the possible receiver
locations on the abdominal surface and cluster them to 16 clusters. Before running that script the `working_directory`
of the `VoxelModel` needs to be set as explained below. 

### Usage
To use the `VoxelModel`, before any model can be loaded the working directory needs to be set. The following
example shows the basic usage:
```python
import os

from LayerModel_lib import VoxelModel
VoxelModel.working_directory = os.path.join('..', 'Phantoms')

# load the voxel model of Donna 
vm = VoxelModel('Donna')
```
