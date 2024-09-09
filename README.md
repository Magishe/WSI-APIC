# WSI-APIC

This repository contains the codes and demonstration data for a Whole Slide Imaging system based on Angular Ptychographic Imaging with a Closed-form Solution (WSI-APIC). It includes GPU-accelerated APIC reconstruction algorithms and sample location segmentation codes.

Paper link: [https://doi.org/10.1364/OPTICA.505283](https://doi.org/10.1364/BOE.538148)

arXiv: [https://arxiv.org/abs/2310.18529](https://www.arxiv.org/pdf/2407.20469)

Original APIC paper link: https://doi.org/10.1038/s41467-024-49126-y

Original APIC code repository: https://github.com/rzcao/APIC-analytical-complex-field-reconstruction

Top-level folder structure:

```bash
├── Data                              # Directory containing raw APIC data and calibration data for preprocessing
├── APIC_Reconstruction.py             # GPU-accelerated APIC reconstruction script for basic image reconstruction
├── APIC_Reconstruction_WholeFOV.py    # GPU-accelerated reconstruction script for full-FOV (2560x2560) images, including auto-stitching functionality
├── subfunctionAPIC                    # Directory for APIC subfunctions used in the reconstruction process
├── Sample location segmentation       # Directory for sample location segmentation
    ├── NSCLC.png                      # Example whole-slide image captured by the sample-locating system
    ├── Sample_Segmentation.py         # Code for sample location segmentation
└── README.md                          # Project documentation (this file)
```


## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [BiBTeX](#BiBTeX)

## Introduction

This repository contains the codes and demonstration data for a Whole Slide Imaging  system based on Angular Ptychographic Imaging with a Closed-form Solution (WSI-APIC). WSI-APIC utilizes Segment Anything model for initial high-level sample location segmentation from a whole slide image, thereby bypassing unnecessary scanning of the background regions and enhancing image acquisition efficiency. A GPU-accelerated APIC algorithm analytically reconstructs phase images with effective digital aberration corrections and improved optical resolutions.

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/Magishe/WSI-APIC.git
    ```

2. Navigate to the project directory:
    ```bash
    cd WSI-APIC
    ```

3. Install the dependencies:
   To set up your environment and install all the necessary packages, run the following command:
    ```bash
    pip3 install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torch-dct==0.1.6 --index-url https://download.pytorch.org/whl/cu121
    pip3 install numpy scipy matplotlib pillow h5py opencv-python torch-dct
    ```


## Usage

### 1. Sample location segmentation
Implement `Sample_Segmentation.py` to automatically locate and segment samples from the image captured by our sample-locating system (`Sample location segmentation/NSCLC.png`).

    python Sample_Segmentation.py
  

### 2. APIC Reconstruction
Implement `APIC_Reconstruction.py` to perform GPU-accelerated APIC reconstruction on small ROI patches.

    python APIC_Reconstruction.py

Tunable Parameters:
#### (1). Dataloading
Assume we want to reconstruct the Siemens Star sample which was imaged using a highly aberrated imaging system, which is inside a folder named "Data". Then, we modify the code as
      
      python APIC_Reconstruction.py --folderName 'Data'

As there is only one file inside the reducedData folder whose name contains "Siemens_Star_g", we can set ```fileNameKeyword``` with name "Siemens_Star_g". If there are multiple files, then we could use ```additionalKeyword```

      python APIC_Reconstruction.py --folderName 'Data' --fileNameKeyword 'Siemens_Star_g'

#### (2). Basic parameters
1. `enableROI`: When it is set to `false`, the program uses the entire field-of-view in the reconstruction. It is recommended to set to `true` as APIC scales badly with respect to the patch size. A good practice is conducting reconstruction using multiple patches and stiching them together to obtain a larger reconstruction coverage.
2. `ROILength`: This parameter is used only when `useROI` is `true`. It specifies the patch sizes used in the reconstruction. It is preferable to set this to be below 256.
3. `ROIcenter`: Define the center of ROI. Example: ROIcenter = [256,256]; ROIcenter = 'auto'.
4. `useAbeCorrection`: Whether to enable aberration correction. It is always recommended to set to `true`. We keep this parameter so that one can see the influence of the aberration if we do not take aberration into consideration.
5. `paddingHighRes`: To generate a high-resolution image, upsampling is typically requried due to the requirement of Nyquist sampling. `paddingHighRes` tells the program the upsampling ratio.

Demo Usage:

      python APIC_Reconstruction.py --enableROI --ROILength 256 --ROIcenter auto --useAbeCorrection --paddingHighRes 3

### 3. APIC Reconstruction for WholeFOV
GPU-accelerated reconstruction script for full-FOV (2560x2560) images, including auto-stitching functionality

    python APIC_Reconstruction_WholeFOV.py

Tunable Parameters:
1. `patchNumber`: The total number of patches into which you intend to divide the full field of view (FOV) along one dimension
2. `overlappingSize`: The overlap size between different patches (for stitching inside one FOV)

Demo Usage:

      python APIC_Reconstruction_WholeFOV.py --patchNumber 5 --overlappingSize 20

## BiBTeX
      @article{Zhao:24,
        author = {Shi Zhao and Haowen Zhou and Siyu (Steven) Lin and Ruizhi Cao and Changhuei Yang},
        journal = {Biomed. Opt. Express},
        keywords = {Image metrics; Imaging systems; Imaging techniques; Laser sources; Phase imaging; Printed circuit boards},
        number = {10},
        pages = {5739--5755},
        publisher = {Optica Publishing Group},
        title = {Efficient, gigapixel-scale, aberration-free whole slide scanner using angular ptychographic imaging with closed-form solution},
        volume = {15},
        month = {Oct},
        year = {2024},
        url = {https://opg.optica.org/boe/abstract.cfm?URI=boe-15-10-5739},
        doi = {10.1364/BOE.538148},
      }

