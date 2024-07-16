from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO(r'D:\code\yolov8\runs\OBB\candy_shangpin\weights\best.pt')

# Tune hyperparameters for 30 epochs
model.tune(data=r'D:\code\yolov8\cfg\candy_shangpin.yaml', 
           epochs=30, 
           iterations=300, 
           optimizer="auto", 
           project='runs/tune',
           name='candy_shangpin',
           plots=False, 
           save=False, 
           val=False)