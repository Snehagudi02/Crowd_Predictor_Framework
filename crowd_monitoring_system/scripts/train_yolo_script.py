from ultralytics import YOLO
import os
import shutil

# Change working directory to project root
if os.getcwd().endswith('notebooks') or os.getcwd().endswith('scripts'):
    os.chdir('..')

os.makedirs('models/cv_weights', exist_ok=True)
yaml_path = os.path.abspath('data/yolo_dataset/crowd_dataset.yaml')

print(f"Using dataset config: {yaml_path}")

# Initialize YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Train the model
# Increasing epochs and using a smaller batch size for CPU stability
print('Starting YOLO training...')
results = model.train(data=yaml_path, epochs=50, imgsz=640, batch=8, device='cpu')

print('Training complete!')

# Copy best weights
best_weights = 'runs/detect/train/weights/best.pt'
dest_path = 'models/cv_weights/yolov8_crowd.pt'

if os.path.exists(best_weights):
    shutil.copy(best_weights, dest_path)
    print(f'Model weights successfully saved to {dest_path}')
else:
    # Try alternate paths if multiple runs
    print('Checking for alternate weight locations...')
    import glob
    weight_files = glob.glob('runs/detect/train*/weights/best.pt')
    if weight_files:
        shutil.copy(weight_files[-1], dest_path)
        print(f'Model weights successfully saved from {weight_files[-1]} to {dest_path}')
    else:
        print('Could not find best.pt. Check ultralytics run directory.')
