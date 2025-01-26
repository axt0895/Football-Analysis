import supervision as sv
import cv2 as cv
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 16
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.25)
            detections.append(detections_batch)
            break
        return detections
        
    def get_object_track(self, frames):
        detections = self.detect_frames(frames)
        for frame_num, detection in enumerate(detections):
            class_names = detection[0].names
            class_names_inv = {v:k for k, v in class_names.items()}
            print(detection)
            try:
                detection_supervision = sv.Detections.from_ultralytics(detection)
                print(detection_supervision)
            except Exception as e:
                print('Error Processing at ', e)