"""
Export the tissue properties from
    A. Fornes-Leal, N. Cardona, M. Frasson, S. Castello-Palacios et al.: Dielectric
    characterization of in vivo abdominal and thoracic tissues in the 0.5 – 26.5 GHz
    frequency band for wireless body area networks. IEEE Access, p. 1, 2019
into a JSON file for the TissueProperties class of LayerModel_lib.
"""
import json
from LayerModel_lib import TissueProperties
from general_functions import json_default

tp = TissueProperties()

i = tp.dielectric_names.index('Muscle')

var_names = ['ef', 'del1', 'tau1', 'alf1', 'del2', 'tau2', 'alf2', 'sig',
             'del3', 'tau3', 'alf3', 'del4', 'tau4', 'alf4']

tissue_equivalance = {'ExternalAir': 'Air',
                      'Air': 'Air',
                      'Aorta': 'Aorta',
                      'Bladder': 'Bladder',
                      'Blood': 'Blood',
                      'BloodVessel': 'Aorta',
                      'BodyFluid': 'Vitreous Humor',
                      'BoneCancellous': 'BoneCancellous',
                      'BoneCortical': 'BoneCortical',
                      'BoneMarrow': 'BoneMarrow',
                      'BrainGreyMatter': 'BrainGreyMatter',
                      'BrainWhiteMatter': 'BrainWhiteMatter',
                      'BreastFat': 'BreastFat',
                      'Cartilage': 'Cartilage',
                      'Cerebellum': 'Cerebellum',
                      'CerebroSpinalFluid': 'CerebroSpinalFluid',
                      'Cervix': 'Cervix',
                      'Colon': 'Colon',
                      'Cornea': 'Cornea',
                      'Duodenum': 'Stomach',
                      'Dura': 'Dura',
                      'EyeSclera': 'EyeSclera',
                      'Fat': 'Fat',
                      'GallBladder': 'GallBladder',
                      'GallBladderBile': 'GallBladderBile',
                      'Gland': 'Thyroid',
                      'Heart': 'Heart',
                      'Kidney': 'Kidney',
                      'Lens': 'Lens',
                      'Liver': 'Liver',
                      'LungDeflated': 'LungDeflated',
                      'LungInflated': 'LungInflated',
                      'Lymph': 'Thyroid',
                      'MucousMembrane': 'SkinWet',
                      'Muscle': 'Muscle',
                      'Nail': 'BoneCortical',
                      'Nerve': 'Nerve',
                      'Oesophagus': 'Stomach',
                      'Ovary': 'Ovary',
                      'Pancreas': 'Thyroid',
                      'Prostate': 'Testis',
                      'Retina': 'EyeSclera',
                      'SkinDry': 'Skin',
                      'SkinWet': 'Skin',
                      'SmallIntestine': 'SmallIntestine',
                      'SpinalCord': 'Nerve',
                      'Spleen': 'Spleen',
                      'Stomach': 'Stomach',
                      'Tendon': 'Tendon',
                      'Testis': 'Testis',
                      'Thymus': 'Thyroid',
                      'Thyroid': 'Thyroid',
                      'Tongue': 'Tongue',
                      'Tooth': 'BoneCortical',
                      'Trachea': 'Trachea',
                      'Uterus': 'Uterus',
                      'Vacuum': 'Vacuum',
                      'Vitreous Humor': 'Vitreous Humor',
                      'LensCortex': 'LensCortex',
                      'LensNucleus': 'LensNucleus',
                      'GIcontents': 'Muscle',
                      'StomachContents': 'Muscle'}

new_values = {'Aorta': [1,	43.25,	9.141,	0.098,	1360.509,	439.155,	0.377,	0.291],
              'Bladder': [2.008,	56.689,	8.203,	0.051,	866.641,	223.742,	0.387,	0.659],
              'Blood': [7.5,	53.673,	8.967,	0.17,	632.735,	5.443,	0.003,	0.027],
              'Colon': [5.575,	57.408,	7.5,	0.142,	1617.041,	21.05,	0.049,	0.24],
              'Oesophagus': [3.492,	58.771,	7.39,	0.113,	934.202,	9.932,	0.057,	0.001],
              'FallopianTubes': [2.022,	53.699,	7.229,	0.007,	391.079,	17.225,	0.267,	0.496],
              'Fat': [5.587, 8.293, 13.5, 0.042, 5.011, 0.435, 0.094, 0.139],
              'GallBladder': [2.849,	55.061,	7.167,	0.072,	264.682,	23.525,	0.312,	0.73],
              'Heart': [2.511,	54.178,	6.914,	0.096,	1161.477,	439.155,	0.4,	0.701],
              'Kidney': [4.85,	51.691,	8.149,	0.122,	452.493,	11.421,	0.136,	0.353],
              'Liver': [6.639,	47.545,	10.329,	0.126,	8.939,	0.375,	0,	0.587],
              'LungDeflated': [6.373,	23.958,	8.569,	0.168,	2000,	58.952,	0.066,	0.089],
              'LungInflated': [6.373,	23.958,	8.569,	0.168,	2000,	58.952,	0.066,	0.089],
              'Muscle': [2.504, 57.81, 8.079, 0.138, 660.06, 7.84, 0.064, 0.103],
              'Ovary': [7.5,	50.358,	8.718,	0.124,	192.885,	2.257,	0.001,	0.029],
              'Pancreas': [7.5,	30.676,	10.045,	0.16,	247.513,	4.687,	0,	0.076],
              'Skin': [2.843,	40.665,	7.725,	0.229,	135.31,	3.072,	0.053,	0.129],
              'SmallIntestine': [6.097,	57.2,	7.777,	0.152,	997.973,	17.63,	0.069,	0.425],
              'Spleen': [6.097,	57.2,	7.777,	0.152,	997.973,	17.63,	0.069,	0.425],
              'Stomach': [4.045,	57.239,	7.641,	0.095,	448.59,	12.308,	0.155,	0.454],
              'Uterus': [1,	55.043,	8.534,	0.029,	2000,	407.5,	0.351,	0.417]  # UterineMatrix
              }


for tissue, values in new_values.items():
    affected_tissues = [orig_t for orig_t, t in tissue_equivalance.items() if t == tissue]
    for tissue_to_change in affected_tissues:
        tissue_index = tp.tissue_names.index(tissue_to_change)
        print(f'Update values for {tissue_to_change}')
        for var_index, var in enumerate(var_names):
            if var_index < len(values):
                if var == 'tau1':
                    value = values[var_index] * 1e-12
                elif var == 'tau2':
                    value = values[var_index] * 1e-9
                else:
                    value = values[var_index]
                tp.values[var][0, tissue_index] = value
            else:
                tp.values[var][0, tissue_index] = 0

data = {'dielectric_names': tp.dielectric_names, 'values': tp.values}
with open('tissue_properties_upv.json', 'w') as fp:
    json.dump(data, fp, default=json_default)