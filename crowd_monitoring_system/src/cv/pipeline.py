import time
import os
import csv
from .detector import PersonDetector
from .tracker import Tracker

class CVPipeline:
    def __init__(self):
        self.detector = PersonDetector()
        self.tracker = Tracker()
        self.csv_path = "data/crowd_data.csv"
        self.last_log_time = time.time()
        self.counts_buffer = []
        
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'count'])

    def process_frame(self, frame):
        boxes = self.detector.detect(frame)
        tracks = self.tracker.update(boxes)
        count = len(tracks)
        
        self.counts_buffer.append(count)
        if len(self.counts_buffer) > 5:
            self.counts_buffer.pop(0)
            
        smoothed_count = int(sum(self.counts_buffer) / len(self.counts_buffer))
        
        current_time = time.time()
        if current_time - self.last_log_time >= 2:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), smoothed_count])
            self.last_log_time = current_time
            
        return {"count": smoothed_count, "tracks": tracks}
