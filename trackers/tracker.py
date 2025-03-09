import supervision as sv
import cv2 as cv
from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
    
    def detect_frames(self, frames):
        batch_size = 6
        detections = []
        for i in range(0, len(frames), batch_size):
            detection = self.model.predict(frames[i: i+batch_size], conf = 0.25)
            detections.append(detection)
            break
        return detections

    def get_object_track(self, frames):
        detections = self.detect_frames(frames)
        
        for detection in detections:
            
            class_names = detection[0].names
            class_names_inv = {v:k for v, k in class_names.items()}
            
            # Convert into supervision detection format
            detection_sv = sv.Detections.from_ultralytics(detection[0])
            
            # Convert into Goalkeeper from Player
            for object_id, class_id in enumerate(detection_sv[0].class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_sv[0].class_id[object_id] = class_names_inv['player']
            
            # Track Objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)
            
            # Convert Goalkeeper into player
            for object_id, class_id in enumerate(detection_with_tracks.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detection_sv.class_id[object_id] = class_names_inv['player']
             
            
            