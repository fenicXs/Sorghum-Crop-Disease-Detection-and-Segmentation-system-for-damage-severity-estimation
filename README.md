# Crop Disease Detection and Segmentation for damage severity estimation

## Crop Disease Detection using YOLOv8 and YOLOv11
### Description
This project focuses on **crop disease detection** leveraging **YOLOv8** and **YOLOv11** for object detection. The goal is to accurately detect and classify diseases in crops based on images. The dataset was annotated using the **Roboflow** platform, and the model was trained and deployed for high-performance inference on custom datasets.

![](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/386f4dc02ee16484932c79bf22a654d36f582070/predict.jpg)

## Key Features
- Utilizes **YOLOv8** and **YOLOv11** models for detection and classification.
- Custom dataset annotation using **Roboflow**.
- End-to-end pipeline:
  - Dataset preparation and annotation.
  - Training using YOLO models.
  - Validation and inference.
- High-resolution detection with configurable confidence thresholds.
- Integration with **Google Colab** for training and deployment.
- Annotated data saving and visualization.

## Installation and Setup

### Prerequisites
1. Python 3.7 or higher.
2. Required libraries:
   - `ultralytics`
   - `roboflow`
   - `torch`
   - `opencv-python`
   - `numpy`
   - `openpyxl`
3. Roboflow API Key (replace the key used here with your key in the code).

### Steps
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/crop-disease-detection.git
   cd crop-disease-detection

2. Install the dependencies:
    ```bash
        pip install -r requirements.txt
3. Download and annotate your dataset on Roboflow.
4. Replace dataset links and API keys in the code with your credentials.

### Dataset
The dataset contains images of crops with varying diseases, labeled and annotated using Roboflow. After annotation, the dataset is exported in YOLO format for compatibility with YOLOv8 and YOLOv11 models.

### Overview
![Overview](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/db5c54086c6dde7a234f4d936e92eac236da5997/dataset%20overview.jpg)

## Dataset Features:
  - Classes: Categories of crop diseases.
  - Images: High-resolution images of crops.
  - Annotations: Bounding boxes indicating diseased areas.

### Training
Training is performed using YOLOv8 and YOLOv11 with the following steps:

  1. Import the dataset:
     ```python
      from roboflow import Roboflow
      rf = Roboflow(api_key="YOUR_API_KEY")
      project = rf.workspace("YOUR_WORKSPACE").project("YOUR_PROJECT")
      version = project.version(2)
      dataset = version.download("yolov8")
  
  2. Start training:
      ```python
      yolo task=detect mode=train model=yolov8m.pt data={dataset.location}/data.yaml epochs=20 imgsz=800 plots=True
  
  3. Evaluate performance using confusion matrices and validation metrics.

### Validation
- Validate the trained model on a test dataset:
  ```python
  yolo task=detect mode=val model={HOME}/runs/detect/train/weights/best.pt data={dataset.location}/data.yaml

### Inference
- Perform inference with the custom-trained model:
  
    ```python
    yolo task=detect mode=predict model={HOME}/runs/detect/train/weights/best.pt conf=0.25 source={dataset.location}/test/images save=True
  
- Save inference results to a Google Drive folder:
  
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    import torch
    
    model = YOLO(f'{HOME}/runs/detect/train/weights/best.pt')
    save_path = os.path.join(f'{HOME}/drive/MyDrive/YOLOv8', 'best.pt')
    torch.save(model.model.state_dict(), save_path)

## Results

  ### Training Metrics:
  - Confusion matrix
  - Precision, Recall, mAP
  - Loss visualization
  
  ### INPUT
  ![Input](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/db5c54086c6dde7a234f4d936e92eac236da5997/input.jpg)
  
  ### OUTPUT
  ![Output](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/db5c54086c6dde7a234f4d936e92eac236da5997/output.jpg)

## Model comparison
![compare](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/10cb716774f34c9e12fd2cfbde78688827c15da6/v8%20vs%20v11%20graph.png)

## Annotated Data Saving
- Save annotated images after inference:
  ```python
  model.predict(source=f'{dataset.location}/test/images', save=True, imgsz=800, conf=0.25)

## Tools and Technologies
  - YOLOv8/YOLOv11: Deep learning object detection models.
  - Roboflow: Dataset annotation and management.
  - Google Colab: Training and testing environment.
  - PyTorch: Backend for YOLO models.
  - OpenCV: Image processing for visualization.

## Future Improvements
  - Implement YOLOv11 enhancements.
  - Add multi-language support for the system.
  - Train models with larger datasets for improved generalization.
---
# **Sorghum Disease Segmentation Using U-Net**

This project focuses on segmenting and estimating the severity of **sorghum rust disease** using **U-Net**, a popular deep learning architecture for semantic segmentation. The solution leverages **ResNet-34** as the encoder and includes extensive data preprocessing and augmentation techniques to improve segmentation accuracy.

## **Project Overview**

### **Key Features**
1. **Disease Segmentation**:  
   Segment affected regions in sorghum crop images to estimate rust disease damage severity.
2. **Preprocessing and Augmentation**:  
   Applied advanced transformations such as sharpening, blurring, and noise addition to enhance the model's generalizability.
3. **Custom Dataset**:  
   Utilized a labeled dataset of sorghum crop images and corresponding binary masks.
4. **Deep Learning Model**:  
   - **Model**: U-Net with ResNet-34 encoder.
   - **Loss Function**: Dice Loss for handling class imbalance.
   - **Metrics**: Intersection over Union (IoU) and F-score for performance evaluation.
5. **Interactive Visualizations**:  
   - Display input images, ground truth masks, and model predictions.
   - Analyze model performance over training epochs using loss and IoU metrics.

## **Dataset**

### **Source**
- The dataset comprises **images of sorghum crops** affected by rust disease along with corresponding binary masks for segmentation.  
- **Image Format**: `.jpg`  
- **Mask Format**: `.png`  

### **Structure**
    file structure
    -Sorghum Disease Image Dataset/
        -Rust/
            -image1.jpg
            -image2.jpg ...
        -Rust_mask/
            -image1.png
            -image2.png 

### **Installation**
Install the required Python libraries

    pip install -U segmentation-models-pytorch albumentations torch torchvision

### Preprocessing and Augmentation
- Transformations:
  - Random Horizontal Flip
  - Resize: 256x256
  - Sharpening
  - Gaussian Noise
  - Normalization: Mean and standard deviation matching ImageNet statistics.

### Model
- Architecture
  - Base Model: U-Net
  - Encoder: ResNet-34 (pretrained on ImageNet)
  - Activation Function: Sigmoid
  - Loss Function: Dice Loss
  - Optimizer: AdamW
  - Learning Rate: 1e-4
- Metrics
  - IoU: Measures overlap between predicted and ground truth masks.
  - F-score: Harmonic mean of precision and recall.
### Training
  - Epochs: 10
  - Batch Size: 64
  - Device: GPU
  - During training:
    - Tracked loss and IoU across epochs.
    - Saved the model with the best IoU score to disk.

### Visualizations
Sample Visualization
![Visual](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/cc4be6ad4887beb2a6f44d3e438ebffbfe5cf8c7/Train%20result.png)

## **Training Performance**

### **IOU Score**
![Visual](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/10cb716774f34c9e12fd2cfbde78688827c15da6/Unet%20IOU.png)

### **DICE LOSS**
![Visual](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/10cb716774f34c9e12fd2cfbde78688827c15da6/Unet%20Loss.png)

### Contact
For questions or collaboration, please contact:
  - Email: pradepkrishna238@gmail.com
  - LinkedIn: https://www.linkedin.com/in/pradeep-k-99a763221
