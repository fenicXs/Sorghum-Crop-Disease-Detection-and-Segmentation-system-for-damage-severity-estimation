# Crop Disease Detection using YOLOv8 and YOLOv11

## Overview
This project focuses on **crop disease detection** leveraging **YOLOv8** and **YOLOv11** for object detection. The goal is to accurately detect and classify diseases in crops based on images. The dataset was annotated using the **Roboflow** platform, and the model was trained and deployed for high-performance inference on custom datasets.

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
3. Roboflow API Key (replace `"kX1T67tvKBnWqTL6ql1I"` with your key in the code).

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

## Overview:
![Overview]()

## Dataset Features:
  - Classes: Categories of crop diseases.
  - Images: High-resolution images of crops.
  - Annotations: Bounding boxes indicating diseased areas.

###Training
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

## Results:
  
  ### Training Metrics:
  - Confusion matrix
  - Precision, Recall, mAP
  - Loss visualization
  
  ### INPUT
  ![Input]()
  
  ### OUTPUT
  ![Output]()

##
### Annotated Data Saving
- Save annotated images after inference:
  ```python
  model.predict(source=f'{dataset.location}/test/images', save=True, imgsz=800, conf=0.25)

### Tools and Technologies
  - YOLOv8/YOLOv11: Deep learning object detection models.
  - Roboflow: Dataset annotation and management.
  - Google Colab: Training and testing environment.
  - PyTorch: Backend for YOLO models.
  - OpenCV: Image processing for visualization.

### Future Improvements
  - Implement YOLOv11 enhancements.
  - Add multi-language support for the system.
  - Train models with larger datasets for improved generalization.

### Contact
For questions or collaboration, please contact:
  - Email: pradepkrishna238@gmail.com
  - LinkedIn: https://www.linkedin.com/in/pradeep-k-99a763221
