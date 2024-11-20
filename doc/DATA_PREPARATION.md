## Datasets
H36M, MPI-INF-3DHP, MPII and TikTok datasets can be downloaded from the offical websites.

Preprocessing of H36M can refer to [this repo](https://github.com/microsoft/multiview-human-pose-estimation-pytorch?tab=readme-ov-file).

Annotations of MPII in JSON format can be found in [this url](https://download.openmmlab.com/mmpose/datasets/mpii_annotations.tar).

The file tree should be like:

```
├── data
│   ├── hm36
│   │   ├── annot
│   │   │   ├── s_01_act_02_subact_01_ca_01
│   │   │   │   ├── matlab_meta.mat
│   │   │   │   ├── matlab_meta.txt
│   │   │   ├── s_01_act_02_subact_01_ca_02
│   │   │   ...
│   │   ├── images
│   ├── mpi_inf_3dhp
│   ├── mpii
│   ├── TikTok_dataset
```

## Masks

We provide masks used in our experiments in [this link](https://drive.google.com/drive/folders/1_EAzT2UI3OTR0mMD9T52yD1I7L0uH5Zv?usp=sharing).

Mask processing code can refer to [this file](https://github.com/Charrrrrlie/segment-anything/blob/main/human_pose/main.py), where we use SAM with 2D keypoints to generate masks.

The file tree should be like:

```
├── data
│   ├── sam_masks
│   │   ├── h36m
│   │   │   ├── s_01_act_02_subact_01_ca_01
│   │   │   │   ├── s_01_act_02_subact_01_ca_01_000001.png
│   │   │   │   ├── ...
│   │   ├── mpi_inf_3dhp
│   │   │   ├── S1
│   │   │   │   ├── Seq1
│   │   │   │   ├── ...
│   │   ├── mpii_val
│   │   │   ├── 000025245.jpg
│   │   │   ├── ...
```

## SMPL models
SMPL models can be downloaded from [this link](https://smpl.is.tue.mpg.de/). 
The regressor `J_regressor_h36m.npy` can be downloaded in [this link](https://github.com/open-mmlab/mmhuman3d/tree/main/configs/hmr/).

(Optional)
The vert segementation file can be found [here](https://raw.githubusercontent.com/Meshcapade/wiki/main/assets/SMPL_body_segmentation/smpl/smpl_vert_segmentation.json)

The file tree should be like:
```
├── data
│   ├── smpl_models
│   │   ├── basicModel_f_lbs_10_207_0_v1.0.0.pkl
│   │   ├── basicmodel_m_lbs_10_207_0_v1.0.0.pkl
│   │   ├── basicModel_neutral_lbs_10_207_0_v1.0.0.pkl
│   │   ├── smpl_vert_segmentation.json
│   │   └── J_regressor_h36m.npy
```


## Synthetic data

### Step1 Data Curation

For simplicty, you can download the SURREAL dataset following the instructions in [this repo](https://github.com/gulvarol/surreal).

You can also process it with our curated distribution by [our reimplementation](https://github.com/Charrrrrlie/surreal). We refactor the code in `custumized_main_part.py` and `custumized_main_part2.py` in a more structured form. The absolute paths in `config` and `run.sh` are required to be modified for your environment.

If successful, you will have the following file tree:
```
├── data
│   ├── surreal
│   │   ├── test
│   │   │   ├── run0
│   │   │   ├── ...
│   │   ├── train
│   │   └── val
│   │
│   ├── surreal_pseudo
│   │   ├── run0_0
│   │   ├── run0_1
│   │   │   ├── ...
```


### Step2 Format Conversion

Download `smpl_webuser` folder from [FLAME](https://github.com/Rubikplayer/flame-fitting/tree/master/smpl_webuser) and place it under `surreal_data_construct` folder.

The SURREAL and our synthetic distribution are commented/uncommented in `surreal_reader.py` main function. You should verify it before processsing.
```
cd surreal_data_construct
python surreal_reader.py
```

If successful, you will have the final file tree, where check_image visualize the projected 3D keypoints and mesh on the image for every 1000 iterations:
```
├── data
│   ├── surreal_h36m_pose
│   │   ├── check_image
│   │   │   ├── check_000000_check.png
│   │   │   ├── ...
│   │   ├── image
│   │   │   ├── image_000000.png
│   │   │   ├── ...
│   │   ├── joints
│   │   ├── mask
│   │   └── info.npy
│   │
│   ├── surreal_h36m_pose_pseudo
│   │   ├── ...
```

#

For the alternative part-segmentaion synthetic data, you can download the code from [this file](https://drive.google.com/drive/folders/1zvW_hgbezWJ2kVlWw3K3g3eLT-3Ohmvj?usp=sharing) and run it on Slurm:

```
./launch_gen.sh <partition> <num_gpu>
```

If successful, you will have the final file tree:
```
├── data
│   ├── surreal_h36m_pose
│   │   ├── image
│   │   │   ├── 0_cam_0_0.png
│   │   │   ├── ...
│   │   ├── joints
│   │   ├── mask
│   │   └── info.npy
```
