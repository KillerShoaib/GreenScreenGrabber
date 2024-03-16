from ultralytics import YOLO

def loadYOLOWorld()->YOLO:
    model =  YOLO('yolov8l-world.pt') # loading the large yolo-world model
    return model

if __name__=='__main__':
    loadYOLOWorld()