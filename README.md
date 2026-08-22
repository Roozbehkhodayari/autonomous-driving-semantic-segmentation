# Autonomous Driving Semantic Segmentation with U-Net

This project uses a **U-Net convolutional neural network** to perform semantic segmentation on autonomous-driving road scenes.

The model receives an RGB road image and predicts a **semantic class for every pixel**, producing a segmentation mask that describes the scene at pixel level.

## Project Goal

The goal is to build an end-to-end semantic segmentation pipeline that can:

- load paired road images and segmentation masks,
- preprocess the data correctly,
- train a U-Net model,
- evaluate performance on a held-out test set,
- analyze class imbalance and class-wise IoU,
- visualize predicted segmentation masks,
- save the trained model for later use.

Semantic segmentation is useful in autonomous-driving applications because it provides more detailed scene understanding than a single image label or a bounding box.

## Dataset

The dataset contains **1,060 paired RGB road-scene images and segmentation masks**.

```text
data/
├── CameraRGB/
└── CameraMask/
```

Each RGB image has a matching segmentation mask with the same filename.

The data is split into:

| Split | Samples | Percentage |
|---|---:|---:|
| Training | 848 | 80% |
| Validation | 106 | 10% |
| Test | 106 | 10% |

A fixed random seed is used so the split can be reproduced.

> The dataset itself is not included in this repository.

## Preprocessing

Images and masks are resized to:

```text
96 × 128
```

The preprocessing pipeline uses different interpolation methods for the two inputs:

- RGB images: **bilinear interpolation**
- Segmentation masks: **nearest-neighbor interpolation**

Nearest-neighbor interpolation is important for masks because the mask contains discrete class IDs and should not create blended class values.

RGB values are converted to floating-point values in the range `0–1`.

The TensorFlow data pipeline also uses:

- caching,
- training-set shuffling,
- batching with a batch size of 32,
- prefetching with `tf.data.AUTOTUNE`.

## U-Net Architecture

The model follows the standard U-Net encoder-decoder structure with skip connections.

```text
Input: 96×128×3

Encoder
96×128×32
     ↓
48×64×64
     ↓
24×32×128
     ↓
12×16×256
     ↓
6×8×512

Decoder
6×8×512
     ↓
12×16×256
     ↓
24×32×128
     ↓
48×64×64
     ↓
96×128×32

Output: 96×128×23
```

The model contains:

- four encoder levels,
- one bottleneck,
- four decoder levels,
- skip connections between matching encoder and decoder stages,
- `Conv2DTranspose` layers for upsampling,
- dropout in deeper layers,
- a final `1×1` convolution for pixel-wise classification.

The network has approximately **8.64 million trainable parameters** and predicts one of **23 semantic classes** for each pixel.

## Training

The model is trained with:

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Output:** logits (`from_logits=True`)
- **Training metric:** pixel accuracy
- **Maximum epochs:** 40
- **Batch size:** 32

Three callbacks are used during training:

- `ModelCheckpoint` to save the model with the best validation loss,
- `ReduceLROnPlateau` to lower the learning rate when validation improvement slows,
- `EarlyStopping` to stop training if validation loss stops improving and restore the best weights.

In this run, validation loss continued improving through epoch 40, so all 40 epochs were completed.

## Results

Performance on the held-out test set:

| Metric | Result |
|---|---:|
| Test Loss | **0.0796** |
| Test Pixel Accuracy | **97.66%** |
| Test Mean IoU | **0.5763** |

At the end of training:

| Metric | Result |
|---|---:|
| Training Accuracy | **97.70%** |
| Validation Accuracy | **97.07%** |
| Best Validation Loss | **0.0981** |

Training and validation performance remained close, so the learning curves did not show a large overfitting gap.

## Class Imbalance Analysis

The dataset is strongly imbalanced.

Classes **7** and **13** together represent about **75.47% of the training pixels**.

Several common classes achieve high IoU:

| Class | IoU |
|---|---:|
| 7 | **0.991** |
| 13 | **0.985** |
| 1 | **0.912** |
| 17 | **0.885** |

Some very rare classes remain much more difficult:

| Class | IoU |
|---|---:|
| 4 | **0.014** |
| 12 | **0.000** |
| 18 | **0.007** |
| 21 | **0.000** |

This explains why pixel accuracy is much higher than Mean IoU. The model performs very well on large and common regions, while performance is less consistent across rare semantic classes.

Class 0 has no pixels in the evaluated data and therefore has an undefined class-wise IoU.

## Qualitative Results

Predictions on held-out test images show that the model captures the overall road-scene structure well.

Large regions are generally segmented accurately, while most remaining errors occur around:

- small regions,
- thin structures,
- fine object boundaries,
- rare semantic classes.

This visual behavior is consistent with the class-wise IoU results.

## Limitations and Possible Improvements

Possible next steps include:

- addressing class imbalance with class-aware weighting or focal loss,
- adding data augmentation while applying the same geometric transformations to images and masks,
- testing a higher input resolution for small objects and fine boundaries,
- extending training to confirm convergence,
- using a sequence-level train/test split if images come from continuous driving sequences.

## Repository Structure

```text
Autonomous-driving-semantic-segmentation/
├── notebooks/
│   └── Unet_Semantic_Segmentation.ipynb
├── images/
├── README.md
├── requirements.txt
└── .gitignore
```

The local `data/` and `artifacts/` folders are excluded from Git because they contain dataset files and trained-model outputs.

## Requirements

Main Python packages used in the notebook:

```text
tensorflow==2.20.0
numpy
matplotlib
imageio
ipykernel
```

Install them with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Clone the repository.
2. Create or activate a Python environment.
3. Install the packages in `requirements.txt`.
4. Place the paired dataset folders under:

```text
data/
├── CameraRGB/
└── CameraMask/
```

5. Open:

```text
notebooks/Unet_Semantic_Segmentation.ipynb
```

6. Run the notebook from top to bottom.

The notebook saves trained models locally under the `artifacts/` directory.

## Main Takeaway

This project demonstrates a complete semantic-segmentation workflow using U-Net: data preparation, encoder-decoder modeling, skip connections, validation-aware training, test evaluation, Mean IoU analysis, class-imbalance investigation, qualitative prediction review, and model saving.

The results show that U-Net can learn the large-scale structure of autonomous-driving road scenes very well, while rare classes and fine details remain the main areas for improvement.
