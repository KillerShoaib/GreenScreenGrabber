import os
import urllib.request
from rich.progress import track
from rich import print


# function to download the weights
def downloadWeights():
    # weights download url
    weights_urls = [
    'https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_cpu.jit',
    'https://huggingface.co/spaces/SkalskiP/YOLO-World/resolve/main/efficient_sam_s_gpu.jit'
    ]   
    folderName = 'EfficientSAMweights'
    print(f"[red]No ESAM weights found[/red]")
    for url in weights_urls:
        filename = url.split('/')[-1]  # Extract the filename from the URL
        filenamePath = os.path.join(folderName,filename)
        
        print(f"[blue]Downloading[/blue]: [green]{filename}[/green]")
        response = urllib.request.urlopen(url)
        total_size = int(response.headers['content-length'])
        
        with open(filenamePath, 'wb') as f:
            for data in track(response, total=total_size):
                f.write(data)

if __name__ =='__main__':
    downloadWeights()

