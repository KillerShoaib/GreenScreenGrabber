import torch
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO
from utils.boxInference import inference_with_boxes
from typing import List
from rich.console import Console
from rich import print
import os


# transparency function
def apply_transparency(
    frame:np.ndarray,
    detections,
    alpha:float=0.75
    )->np.ndarray:
    

    # Extract masks from detections (assuming detections structure provides access to masks)
    masks = detections.mask

    # Combine masks (logical OR operation)
    combined_mask = np.any(masks, axis=0).astype(np.uint8)  # Ensure byte data type

   # Create an RGBA image with a black background and full opacity
    height, width, _ = frame.shape  # Get image dimensions
    background = np.zeros((height, width, 4), dtype=np.uint8)
    background[:, :, 0:3] = frame  # Copy BGR channels from original frame
    background[:, :, 3] = 255  # Set alpha channel to 255 (fully opaque)

    # Invert the combined mask to get the transparent areas
    transparent_mask = cv2.bitwise_not(combined_mask)

    # Use weighted averaging for smooth blending
    foreground = cv2.bitwise_and(background, background, mask=combined_mask)
    background = cv2.bitwise_and(background, background, mask=transparent_mask)
    result = cv2.addWeighted(foreground, alpha, background, 1 - alpha, 0)

    return result.astype(np.uint8)


def RemoveImgBG(
    input_image: np.ndarray,
    categories: List[str],
    device:torch.device,
    yolo_world_model:YOLO,
    efficient_sam_model:torch.jit.ScriptFunction,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_class_agnostic_nms: bool = False,
) -> np.ndarray:

    # setting a wait spinner
    console = Console()
    with console.status("[bold green]Removing Image Background...[/bold green] :hourglass:",spinner='aesthetic') as status:

        # set classes for yoloworld model
        yolo_world_model.set_classes(categories)
        # predict
        results = yolo_world_model.predict(input_image, conf=confidence_threshold,verbose=False)

        # getting the detections from the results
        detections = sv.Detections.from_ultralytics(results[0])

        # checking if bbox exist or not and returning None
        if (detections.xyxy.size==0):
            return np.array([]) # returning empty array

        # using nms
        detections = detections.with_nms(
            class_agnostic=with_class_agnostic_nms,
            threshold=iou_threshold
        )
        
        detections.mask = inference_with_boxes(
            image=input_image,
            xyxy=detections.xyxy,
            model=efficient_sam_model,
            device=device
        )

        # apply transparency fucnction
        frame = input_image
        frame = apply_transparency(frame=frame,
                            detections=detections,
                            alpha=1)



        return frame

# a function to save the image
def saveImg(img:np.ndarray,filename:str)->None:
    root_save_img_dir = 'outputImages'
    imgName,_ = os.path.splitext(filename)
    ext = ".png"
    counter=0 # to add with the image name

    # creating the directory if don't exist
    if not (os.path.exists(root_save_img_dir)):
        try:
            os.mkdir(root_save_img_dir)
        except OSError:
            console = Console()
            console.print(f"Creation of the directory {root_save_img_dir} failed",style="bold black on red")


    # looping over to check if the same name exist or not
    while True:
        # Generate new filename
        filename = f"{imgName}{counter}{ext}"
        file_path = os.path.join(root_save_img_dir, filename)

        # Check if file already exists
        if not os.path.exists(file_path):
            cv2.imwrite(file_path,img)
            console = Console()
            console.print(f"📂 Image saved as {file_path}",style="bold black on green")
            break
        else:
            counter += 1

    return None

def imgProcess(
    input_image_path: str,
    categories: List[str],
    device:torch.device,
    yolo_world_model:YOLO,
    efficient_sam_model:torch.jit.ScriptFunction,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.5,
    with_class_agnostic_nms: bool = False,
)->None:
    # extract the image name only
    image_name = input_image_path.split('/')[-1]

    # reading the image
    input_image = cv2.imread(input_image_path)

    # calling the function for removedImg

    ImgBGRemoved = RemoveImgBG(
        input_image=input_image,
        categories=categories,
        device=device,
        yolo_world_model=yolo_world_model,
        efficient_sam_model=efficient_sam_model,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        with_class_agnostic_nms=with_class_agnostic_nms
    )

    # checking if there is object of interest
    if(ImgBGRemoved.size == 0):
        console = Console()
        console = Console()
        console.print("❗Unable to find the desired object in the Image :pensive:",style="bold black on red")
        console.print("💡 Tips: Try Different category name, try to reduce or increase -iou -conf values.",style="bold black on yellow")
        return None
    else:
        # Calling the saveImg function to save the image
        saveImg(img=ImgBGRemoved,filename=image_name)
        return None
