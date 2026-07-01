#TODO Install Python 3.12
#winget install Python.Python.3.12
#TODO Install OpenCV untuk webcam
#pip install torch torch-vision opencv-python
#TODO Container
#python -m venv yolov8-env yolov8-env\Scripts\activate
#TODO Upgrade Package
#pip install –upgrade pip
#TODO Install YOLO
#pip install ultralytics
#TODO Install Pandas
#pip install pandas
#TODO Web Library
#pip install flask flask-cors
#TODO 
#pip install pyttsx3
#TODO Train
#

# from ultralytics import YOLO

# if __name__ == '__main__':
#     model = YOLO('yolov8n-pose.pt')

#     model.train(
#         data='C:/Users/ACER/Videos/ComputerVision/dataset/data.yaml', 
#         epochs=100,          
#         imgsz=640,
#         patience=15,        
        
#         mosaic=1.0,         
#         degrees=10.0,    
#         fliplr=0.5         
#     )

# from ultralytics import YOLO

# if __name__ == "__main__":
#     model = YOLO("yolov8n-cls.pt")

#     model.train(
#         data="C:/Users/ACER/Videos/ComputerVision/dataset_2",
#         epochs=80,
#         imgsz=224,
#         batch=16,
#         patience=20,

#         augment=True
#     )