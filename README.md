# NeRF<sup>*</sup>(3D Reconstruction Model)를 이용한 Data Augmentation기법의 Image Classification 성능 탐구 
> A Study on the Image Classification Performance of Data Augmentation Techniques Using NeRF<sup>*</sup>(3D Reconstruction Model)

> 2024 FALL AI Intensive Class1 (SCE3319, F135-1) Project

## 🚩 Table of Contents

- [Project summary](#-project-summary)
- [Project structure](#-project-structure)
- [Methods](#-methods)
- [Results](#-results)

## 📝 Project summary

### A Study on the Image Classification Performance of Data Augmentation Techniques Using NeRF<sup>*</sup>(3D Reconstruction Model)

- The project aims to utilize 3D object generation models like NeRF, 3D Gaussian Splatting, or Mesh to better understand occluded or unseen parts of objects, such as their backsides, and to address the issue of viewpoint variation in image classification.
- We initiated the project with the belief that these models can enhance image classification accuracy by generating diverse viewpoints.

### Team member

| Dept     | Icon                                                                                                                                     | Name          | Github                                                                                                                                   |
| -------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| software | <img src="https://avatars.githubusercontent.com/u/38002846?v=4" width="50">                                                              | Minchang Kim | [<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>](https://github.com/minchang-KIm) |
| software | <img src="https://github.com/Data-Mining-AI-Paper/DATA_MINING_AI_PAPER/assets/78012131/e9bf5d98-277a-492f-a6f5-924f41c8ce67" width="50"> | Jongho Baik   | [<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=GitHub&logoColor=white"/>](https://github.com/JongHoB)    |

## 🏗️ Project structure

### Datasets

- Click the image to see the datasets.
  
[<img src="https://github.com/user-attachments/assets/2cd6f151-439c-40be-bda6-c731f0279a24" width="300"/>](https://drive.google.com/drive/folders/11uGebvTBNHXhFe_3MCy2rR-CTfqGFIgH?usp=sharing)

```bash
├── 1. train_label_baseline.csv
├── 2. train_label_3d.csv
├── 3. test_label.csv
├── 4. best_model_baseline.pth
├── 5. best_model_3d.pth
└── datasets
      ├── 3D images
      │      └── dataset6_batch*_output.zip
      └── dataset*.zip
```

- *The Google Drive URL will be expired soon because of Storage Limitation and ShapeNetCore Datasets Licenses.*

### Directory

```bash
/AI-1
├── 1. 3D-ResNet18.ipynb
├── 2. Baseline-ResNet18.ipynb
├── 3. dataloaders_ShapeNetCore.ipynb
└── README.md
```

### Details

- `train_label_baseline.csv`: Training labels for the baseline models.
- `train_label_3d.csv`: Training labels including images generated using InstantMesh.
- `test_label.csv`: Test labels.
- `best_model_baseline.pth`: Weights of the best baseline model.
- `best_model_3d.pth`: Weights of the best model training with images generated using InstantMesh.
- `dataset6_batch*_output.zip`: Images generated using InstantMesh. The original image is from point of view 6.
- `dataset*.zip`: Original datasets, including training and test datasets. Each number corresponds to a point of view.
---
- `3D-ResNet18.ipynb`
- `3D_ObjectGen.py` :
- `Capture_Image_from_Object.py` :
- `Baseline-ResNet18.ipynb`
- `dataloaders_ShapeNetCore.ipynb`:  Images generated from the ShapeNetCore dataset.

## 🔨 Methods

### Model

- **[InstantMesh](https://github.com/TencentARC/InstantMesh)**
  - Goals:
    1. Fast generation from a single image.
    2. Applicable to various categories, not just for car and chair categories.
  
---

### Point of View

- In this project, we use 14 povs as follows.

| **Filename**    | **Perspective**          |
|-----------------|--------------------------|
| filename_0.png  | Front                    |
| filename_1.png  | Back                     |
| filename_2.png  | Left Side                |
| filename_3.png  | Right Side               |
| filename_4.png  | Top                      |
| filename_5.png  | Top Left                 |
| filename_6.png  | Top Right                |
| filename_7.png  | (Back) Top Right         |
| filename_8.png  | (Back) Top Left          |
| filename_9.png  | Bottom                   |
| filename_10.png | Bottom Left              |
| filename_11.png | Bottom Right             |
| filename_12.png | (Back) Bottom Right      |
| filename_13.png | (Back) Bottom Left       |

- When using InstantMesh, we use `filename_6.png` (Top Right perspective) because it represents at least three planes, providing more detailed information about the object.

    <img src="https://github.com/user-attachments/assets/c49f8c81-8477-4a27-9135-8f61e672e6ff" width="200">

  
---

### Classes
- The original ShapeNetCore dataset categories are identified by `synset_id`. We have mapped these to custom-defined indices as shown below:

| **synset_id** | **category**   | **index** |
|---------------|----------------|-----------|
| 2691156       | airplane       | 0         |
| 2747177       | trash bin      | 1         |
| 2773838       | bag            | 2         |
| 2801938       | basket         | 3         |
| 2808440       | bathtub        | 4         |
| 2818832       | bed            | 5         |
| 2828884       | bench          | 6         |
| 2843684       | birdhouse      | 7         |
| 2871439       | bookshelf      | 8         |
| 2876657       | bottle         | 9         |
| 2880940       | bowl           | 10        |
| 2924116       | bus            | 11        |
| 2933112       | cabinet        | 12        |
| 2942699       | camera         | 13        |
| 2946921       | can            | 14        |
| 2954340       | cap            | 15        |
| 2958343       | car            | 16        |
| 2992529       | cellphone      | 17        |
| 3001627       | chair          | 18        |
| 3046257       | clock          | 19        |
| 3085013       | keyboard       | 20        |
| 3207941       | dishwasher     | 21        |
| 3211117       | display        | 22        |
| 3261776       | earphone       | 23        |
| 3325088       | faucet         | 24        |
| 3337140       | file cabinet   | 25        |
| 3467517       | guitar         | 26        |
| 3513137       | helmet         | 27        |
| 3593526       | jar            | 28        |
| 3624134       | knife          | 29        |
| 3636649       | lamp           | 30        |
| 3642806       | laptop         | 31        |
| 3691459       | loudspeaker    | 32        |
| 3710193       | mailbox        | 33        |
| 3759954       | microphone     | 34        |
| 3761084       | microwave      | 35        |
| 3790512       | motorbike      | 36        |
| 3797390       | mug            | 37        |
| 3928116       | piano          | 38        |
| 3938244       | pillow         | 39        |
| 3948459       | pistol         | 40        |
| 3991062       | pot            | 41        |
| 4004475       | printer        | 42        |
| 4074963       | remote         | 43        |
| 4090263       | rifle          | 44        |
| 4099429       | rocket         | 45        |
|4225987     		| skateboard   	|46         	|
|4256520     		| sofa         		|47         	|
|4330267     		| stove        		|48         	|
|4379243     		| table        		|49         	|
|4401088     		| telephone    	|50         	|
|4460130     		| tower        		|51         	|
|4468005     		| train        		|52         	|
|4530566     		| watercraft   	|53         	|
|4554684     		| washer      		|54         	|

---

## 📊 Results

| | **Baseline Model - ResNet18**   | **Baseline + 3D generated Images** |
|---------------|----------------|-----------|
| Accuracy       | 77.2%       | 78.9%         |


---

## 🔎 Limitations

- As we can see in Github Issues in Pytorch3D (https://github.com/facebookresearch/pytorch3d/issues/666, https://github.com/facebookresearch/pytorch3d/issues/313,...), ***ShapeNetCore datasets are far from good qualities.***
  - It seems that each object needs fine-tuning (lighting, texture, etc.).
    - However, ShapeNetCore contains more than 50,000 objects...
  -  **The quality of the image itself surely affects the process of generating 3D objects in InstantMesh.**
- We used 12 Colab sessions simultaneously for this project.
  - The biggest weakness might be the limited number of experiments.
