# Crop Disease Detection and Segmentation for damage severity estimation

## Crop Disease Detection using YOLOv8 and YOLOv11
### Description
This project focuses on **crop disease detection** leveraging **YOLOv8** and **YOLOv11** for object detection. The goal is to accurately detect and classify diseases in crops based on images. The dataset was annotated using the **Roboflow** platform, and the model was trained and deployed for high-performance inference on custom datasets.

![](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/386f4dc02ee16484932c79bf22a654d36f582070/predict.jpg)

### Dataset
The dataset contains images of crops with varying diseases, labeled and annotated using Roboflow. After annotation, the dataset is exported in YOLO format for compatibility with YOLOv8 and YOLOv11 models.

### Overview
![Overview](https://github.com/fenicXs/Crop-Disease-Detection-system/blob/db5c54086c6dde7a234f4d936e92eac236da5997/dataset%20overview.jpg)
  
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
# **Sorghum Disease Segmentation**

## **U-net**
This project focuses on segmenting and estimating the severity of **sorghum rust disease** using **U-Net**, a popular deep learning architecture for semantic segmentation. The solution leverages **ResNet-34** as the encoder and includes extensive data preprocessing and augmentation techniques to improve segmentation accuracy.

## **Project Overview**

### **Sample Visualization**
![Visual](https://github.com/fenicXs/Sorghum-Crop-Disease-Detection-and-Segmentation-system-for-damage-severity-estimation/blob/cc4be6ad4887beb2a6f44d3e438ebffbfe5cf8c7/Train%20result.png)


### Contact
For questions or collaboration, please contact:
  - Email: pradepkrishna238@gmail.com
  - LinkedIn: https://www.linkedin.com/in/pradeep-k-99a763221
