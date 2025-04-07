# **Sorghum Disease Segmentation**

## **U-net**
Implements a U-Net-based semantic segmentation model to identify and segment diseased regions on sorghum leaves from image data.

### **Key Features**
- U-Net architecture with encoder-decoder structure for pixel-wise segmentation.
- Trained using *binary_crossentropy* loss for binary mask prediction.
- Input images are resized to 512×768 and normalized for consistent training.
- Includes performance metrics like Accuracy, Precision, Recall, and MeanIoU.
- Implements training optimizations such as Early Stopping, Learning Rate Reduction, and Model Checkpointing.
- Efficient data loading pipeline using *tf.data.Dataset* and OpenCV preprocessing.

![Unet](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/4d1343c23e1e57d8ac24820ff2ee1ea4a9b549d2/Segmentation/Unet.png)

## **Residual U-net**
The solution leverages **ResNet-34** as the encoder and includes extensive data preprocessing and augmentation techniques to improve segmentation accuracy.

### **Key Features**

- Encoder: ResNet-34 (no pre-trained weights)
- Input Size: 256x256
- Loss Function: Dice Loss
- Optimizer: *AdamW*
- Normalization using ImageNet mean and std

![ResUnet](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/cc4be6ad4887beb2a6f44d3e438ebffbfe5cf8c7/Train%20result.png)

## **MultiResidual U-net**
MultiResUNet is a powerful architecture designed for robust segmentation, especially on noisy, complex datasets. It addresses two limitations of the traditional UNet:

1. Multi-resolution filters in each encoder block extract diverse spatial features.
2. Residual paths (ResPaths) maintain feature identity and reduce degradation during deeper network traversal.

### **Key Features**

- MultiResBlock: Applies 3×3, 5×5, and 7×7 convolutions in parallel with residual connections.
- ResPath: Series of convolutional blocks maintaining long-term feature propagation.
- Encoder-Decoder: Classic U-Net style with skip connections and transposed convolutions for upsampling.
- Final Activation: sigmoid for binary segmentation tasks.
- Input Shape: (height, width, channels) (e.g., (256, 192, 3) for RGB images)

![MRU](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/4d1343c23e1e57d8ac24820ff2ee1ea4a9b549d2/Segmentation/MRU.png)

## **NVIDIA Segformer**
This project leverages the SegFormer architecture from NVIDIA to perform semantic segmentation on sorghum leaf images affected by rust disease. The dataset is sourced from Roboflow, annotated using PNG masks for pixel-level segmentation.

### Dataset
- Source: Roboflow
- Format: Semantic segmentation (.jpg images with .png masks)
- Classes: Derived from _classes.csv in Roboflow export
- Structure:
``` bash
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1_mask.png
|   └── _classes.csv
├── valid/
└── test/
```
### **Key Features**

- Base Model: nvidia/segformer-b0-finetuned-ade-512-512
- Transformer-based lightweight backbone (MiT-B0)
- Supports fine-tuning on custom datasets
- Data Preprocessing via SegformerFeatureExtractor (resizing, normalization, label alignment)

![Segformer](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/4d1343c23e1e57d8ac24820ff2ee1ea4a9b549d2/Segmentation/Segformer.png)
