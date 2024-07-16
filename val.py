from ultralytics import YOLO

model = YOLO('D:\yolov8\\runs\OBB\kebike\weights\\best.pt')
# Run batched inference on a list of images
results = model('D:\yolov8\guaTiao\\kebike\images\\train/0e071fc1-3e08-4b18-9564-2f470b5ae424.jpg')  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk