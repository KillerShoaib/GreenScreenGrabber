import os
import cv2
import torch
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
import argparse
from rich_argparse import RichHelpFormatter
import sys
from utils import (loadEfficientSAM,loadYoloWorld,
                    imgProcessor,vidProcessor)


# setting the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# getting the models
efficient_sam_model = loadEfficientSAM.loadSAM(device=DEVICE)
yoloWorld_model = loadYoloWorld.loadYOLOWorld()


def showTable(
    path:str,
    categories:str,
    confidence_threshold:float,
    iou_threshold:float,
    class_agnostic_nms:bool
):

    table = Table(title="Green Screen Creator Argument Values")

    table.add_column("Path", justify="right", style="cyan", no_wrap=True)
    table.add_column("Categories", style="magenta")
    table.add_column("Confidence Threshold", justify="right", style="green")
    table.add_column("IoU Threshold", justify="right", style="blue")
    table.add_column("Class Agnostic NMS", justify="right", style="yellow")

    table.add_row(f"{path}", f"{categories}", f"{confidence_threshold}", f"{iou_threshold}", f"{class_agnostic_nms}")
    console = Console()
    console.print("💡 Tips: If the desired object is unable to detect try to use different category name, play with different values of -iou -conf",style="bold black on yellow")
    console.print(table)
    


# validation functions

# validate iou, confidence value
def validate_confidence(value:float)->None:
    if not 0 <= value <= 1:
        console = Console()
        console.print(f"❗Error: Invalid confidence level ({value}) provided (-c/--confidence). Please enter a value between 0 (low) and 1 (high).",style="bold black on red")
        sys.exit()
    return None

def validate_iou(value:float)->None:
    if not 0 <= value <= 1:
        console = Console()
        console.print(f"❗Error: Invalid IoU level ({value}) provided (-iou/--iou). Please enter a value between 0 (low) and 1 (high).",style="bold black on red")
        sys.exit()
    return None

# validate the categories input
def validate_category_list(value:str)->None:
    if not isinstance(value, str):
        console =Console()
        console.print(f"❗Error: Invalid Data Type. (-c,--catogories) value must be string.",style="bold black on red")
        sys.exit()
    if not value:
        console = Console()
        console.print(f"❗Error: (-c,--catogories) can not have empty string.",style="bold black on red")
        sys.exit()

    return None



# check for img or video
def checkImgVid(path:str)->str:
    _,ext = os.path.splitext(path)

    image_ext = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
    video_ext = [".mp4", ".avi", ".mkv", ".mov"]

    if ext in image_ext:
        return 'img'
    elif ext in video_ext:
        return 'vid'
    else:
        return 'unidentified'


if __name__=='__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Remove Image/Video Background using text prompt.",
                                    formatter_class=RichHelpFormatter)

    # Required argument
    parser.add_argument("path", type=str, help="Path to the image or video file.")

    # Optional arguments with defaults
    parser.add_argument("-c", "--category_list", type=str, nargs='?',default=["person"],
                        help="Comma separated list of categories to detect. (default: ['person'])")
    parser.add_argument("-conf", "--confidence", type=float, default=0.5,
                        help="Minimum confidence threshold (default: 0.5).")
    parser.add_argument("-iou", "--iou", type=float, default=0.4,
                        help="Minimum Intersection-over-Union threshold (default: 0.4).")
    parser.add_argument("-nms", "--nms_agnostic", action="store_true", default=False,
                        help="Enable Class Agnostic NMS (default: False).")
    # Parse arguments
    args = parser.parse_args()

    # getting all the values
    path = args.path
    
    # catergories in string format
    categories = args.category_list

    # float and bool values
    confidence_threshold = args.confidence
    iou_threshold = args.iou
    agn_nms = args.nms_agnostic

    # validating inputs
    validate_confidence(confidence_threshold)
    validate_iou(iou_threshold)
    validate_category_list(categories)
    


    # displaying argument table
    showTable(
        path=path,
        categories=categories,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        class_agnostic_nms=agn_nms
    )

    #converting categories in list
    # Split the string using the comma (",") as a delimiter
    categories = categories.split(",")
    categories = [item.strip() for item in categories]

    checkExt = checkImgVid(path=path) # checking if the path is a image or not

    # running imgProcess for img and vidProcess for video
    if(checkExt=='img'):
        imgProcessor.imgProcess(
            input_image_path=path,
            categories=categories,
            device=DEVICE,
            yolo_world_model=yoloWorld_model,
            efficient_sam_model=efficient_sam_model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            with_class_agnostic_nms=agn_nms
        )
    elif(checkExt=='vid'):
        vidProcessor.vidProcess(
            input_video_path=path,
            categories=categories,
            device=DEVICE,
            yolo_world_model=yoloWorld_model,
            efficient_sam_model=efficient_sam_model,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            with_class_agnostic_nms=agn_nms
        )
    else:
        console = Console()
        console.print("❗Error: The file at the provided path is not a valid image or video file. Please check the file extension (e.g., .jpg, .mp4)",style="bold black on red")