import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

from utils.SAMDownloader import downloadWeights
import torch
import numpy as np
import os
from rich import print

def checkDownload():

    filenames = ['EfficientSAMweights/efficient_sam_s_cpu.jit',
    'EfficientSAMweights/efficient_sam_s_gpu.jit']
    for filename in filenames:
        if not os.path.exists(filename):
            try:
                os.mkdir('EfficientSAMweights')
            except OSError:
                print(f"Creation of the directory EfficientSAMweights failed")
            downloadWeights() # downloading the weights
            break


def loadSAM(device: torch.device) -> torch.jit.ScriptFunction:
    # defining the checkpoints name
    CPU_SAM_CHECKPOINT = 'EfficientSAMweights/efficient_sam_s_cpu.jit'
    GPU_SAM_CHECKPOINT = 'EfficientSAMweights/efficient_sam_s_gpu.jit'

    # checking if the weights already exist or not
    checkDownload()

    # now load the model
    if device.type=='cuda':
        model = torch.jit.load(GPU_SAM_CHECKPOINT)
    else:
        model = torch.jit.load(CPU_SAM_CHECKPOINT)
    
    # returning the model
    return model


if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = loadSAM(DEVICE)
    print(type(model))