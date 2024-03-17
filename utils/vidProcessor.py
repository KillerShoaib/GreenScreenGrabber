import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

import torch
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from utils.boxInference import inference_with_boxes
from typing import List
from rich.progress import track
from rich.text import Text
from rich.console import Console
from rich import print
import os


def RemoveVidBG(
  input_video: str,
  target_video_path: str,
  categories: List[str],
  device:torch.device,
  yolo_world_model:YOLO,
  efficient_sam_model:torch.jit.ScriptFunction,
  confidence_threshold: float = 0.3,
  iou_threshold: float = 0.5,
  with_class_agnostic_nms: bool = False,
) -> None:

  # creating a frame generator using supervision
  frame_generator = sv.get_video_frames_generator(input_video)
  video_info = sv.VideoInfo.from_video_path(input_video)

  # getting the total video area
  width, height = video_info.resolution_wh
  frame_area = width * height

  # setting the categories
  yolo_world_model.set_classes(categories)

  # for cli logging purpose
  progressText = Text("Creating Green Screen :hourglass:")
  
  with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
  
    for frame in track(frame_generator, total=video_info.total_frames, description=progressText): # reading using sv generator

      results = yolo_world_model.predict(frame, conf=confidence_threshold,verbose=False)
      detections = sv.Detections.from_ultralytics(results[0])
      detections = detections.with_nms(
          class_agnostic=with_class_agnostic_nms,
          threshold=iou_threshold
      )
      
        # checking if the bbox exist or not
      if(detections.xyxy.size==0):
        sink.write_frame(frame) # writing the original frame as it is
        continue


      
      detections.mask = inference_with_boxes(
          image=frame,
          xyxy=detections.xyxy,
          model=efficient_sam_model,
          device=device
      )

      # BG remove to greem
      frame_cpy = frame.copy()
      combined_mask = np.any(detections.mask,axis=0)
      frame_cpy[~combined_mask] = [0, 255, 0]  # Apply the combined mask

      sink.write_frame(frame_cpy)
  
  console = Console()
  console.print(f"📂 Video saved at {target_video_path} ",style="bold black on green")
  

def targetVidPathGenerator(input_video_path:str)->str:
    root_save_vid_dir = 'outputVideos'
    videoPath,ext = os.path.splitext(input_video_path)
    videoName = videoPath.split('/')[-1]
    counter=0

    # creating the directory if don't exist
    if not (os.path.exists(root_save_vid_dir)):
        try:
          os.mkdir(root_save_vid_dir)
        except OSError:
          console =Console()
          console.print(f"⛔ Creation of the directory {root_save_vid_dir} failed",style="bold black on red")
    # looping over to check if the same name exist or not
    while True:
        # Generate new filename
        filename = f"{videoName}GreenScreen{counter}{ext}"
        file_path = os.path.join(root_save_vid_dir, filename)

        # Check if file already exists
        if not os.path.exists(file_path):
            return file_path
        else:
            counter += 1
    return ''

def vidProcess(
  input_video_path: str,
  categories: List[str],
  device:torch.device,
  yolo_world_model:YOLO,
  efficient_sam_model:torch.jit.ScriptFunction,
  confidence_threshold: float = 0.3,
  iou_threshold: float = 0.5,
  with_class_agnostic_nms: bool = False,
)->None:

  # generating target_video path
  target_vid_path = targetVidPathGenerator(input_video_path)

  # calling the function to create the green screen
  RemoveVidBG(
        input_video=input_video_path,
        target_video_path=target_vid_path,
        categories=categories,
        device=device,
        yolo_world_model=yolo_world_model,
        efficient_sam_model=efficient_sam_model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        with_class_agnostic_nms=with_class_agnostic_nms
    )