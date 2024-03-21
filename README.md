![GreenScreenGrabber Github Headline](https://github.com/KillerShoaib/GreenScreenGrabber/assets/59968346/48725e5f-2858-468f-a2f4-1f2cb4945e5b)
<hr/>

## Table of contents

1. [**Overview**](#overview)
2. [**Demo**](#demo)
3. [**Run Locally**](#runlocally)
4. [**Usage**](#usage)
5. [**Example**](#example)
6. [**Input & Output Path**](#input-output)
7. [**Google Colab Notebook**](#notebook)
8. [**References**](#ref)
9. [**Contribution**](#cont)
10. [**License**](#lic)

<h1 id="overview">
    <div align="center">Overview</div>
</h1>
<div align="center">
    <b>Green Screen Grabber</b> is a Python <b>CLI</b> application designed to create <b>green screens</b> on videos or <b>transparent images</b> based on the object or <b>objects specified as text prompts</b> (e.g: "person, cat"). It <b>detects</b> the given object, <b>masks</b> it, and creates a green screen around that masked object or makes the image around the masked object transparent.<br><br>

</div>

<div align="center">
    This application is built using a <b>combination of two of the best state-of-the-art open-source models</b> available to date. For <b>text-based object detection</b>, I've utilized <b><code>YOLO World</code></b>, an open vocabulary object detection model that is <b>significantly faster</b> than <b><code>Grounding Dino V2</code></b>. For segmentation, I've employed <b><code>Efficient SAM</code></b>, a variant of <b><code>SAM</code></b> (Segment Anything Model), which is <b>4x faster than the original SAM</b>. <br><br>
</div>

<h1 id="demo">
    <div align="center">Demo</div>
</h1>


## Transparent Image Example
![ImagesExample](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExc3gydDMwa3ljMmpkeWoxeGR2cnB5ZnJzcHR1a3N0aTRneW53cW1kdCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/ycFihVyFGYmBdmFdvf/giphy.gif)

## Green Screen Video Example
![GreenScreenGrabber Github Headline](https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExcjh1eDVsYnZnMm4ya2RuajZxM2VwMHdhNjQ5bWEzYzNrenJhM3lzNCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/g22PcpPc3iFQ8z9Tnv/giphy.gif)


<h1 id="runlocally">
    <div align="center">Run Locally</div>
</h1>

> **Note:** To generate green screen video it **reuires T4 or equivalent GPU**. Otherwise it'll be painfuly slow.

- **Clone the repository**

```bash
git clone https://github.com/KillerShoaib/GreenScreenGrabber.git
```


- **Change the directory to the project directory**

```bash
cd GreenScreenGrabber
```

- **Create a virtual environment and activate it**

```bash
python3 -m venv venv
source venv/bin/activate
```
> **Note:** Shown for Linux Distro Only:

- **Install all the dependencies.**

```bash
pip install -r requirements.txt
```

**Prerequisite**
- Python 3.8 or above

<br>

<h1 id="usage">
    <div align="center">Usage</div>
</h1>


**Run the program from the command line**

```bash
python3 RemoveBG.py <Image or video path> -c <category list> -conf <confidence value> -iou <iou value> -nms
```

**For detailed usage instructions, refer to the [Argument Details](#argument-details).**

## Argument Details

Here's a list of available command line arguments.

| Argument | Value Type | Argument Type | Description | Default | Required |
|----------|------------|---------------|-------------|---------|----------|
|      | Image or Video path       | positional    | Takes the path of the image or video |   None  | Yes      |
| `-c` , `--category`     | string      | optional      | Takes category list with comma separated value. i.e: "person,cat" | "Person"   | No       |
| `-conf` , `--confidence`     | float      | optional      | Takes the minimum confidence value for the object to detect | 0.5   | No       |
| `-iou` , `--iou`     | float      | optional      | Takes the Intersection Over Union Value | 0.4   | No       |
| `-nms` , `--nms_agnostic`     | boolean      | optional      | Sets Class Agnostic Non Maximum Suppression value. Just flag the argument to set it True. | False   | No       |
| `-h` , `--help`     | None      | optional      | Shows all the argument details in the terminal   | None      | No

<h1 id="example">
    <div align="center">Example</div>
</h1>

```bash
python3 RemoveBG.py Images/image.png -c "Cartoon,Hair" -conf 0.001 -iou 0.1
```

> **Note:** Where image.png is inside Images folder. And Images folder is in the root directory of the project.

<h1 id="input-output">
    <div align="center">Input & Output Path</div>
</h1>

- **Input path** : Place the image or video directly in the project's root directory (the main folder). Or give the total path of the image/video.

- **Output path** : All the output images and videos will inside `outputImages` and `outputVideos` folder of the root directory.

<h1 id="notebook">
    <div align="center">Google Colab Notebook</div>
</h1>

If you don't have a GPU then you can use this [**Google colab notebook**](https://colab.research.google.com/drive/1W3b3jne7C9s6_ru3YTs2teDK1Zv8WV__?usp=sharing) to run the program. Head over to the notebook and follow the instruction given in the notebook.



<h1 id="ref">
    <div align="center">References</div>
</h1>


- [**YOLO World Paper**](https://arxiv.org/abs/2401.17270)
- [**Efficient SAM Paper**](https://arxiv.org/abs/2312.00863)
- [**YOLO-World Huggingface Space by SkalskiP**](https://huggingface.co/spaces/SkalskiP/YOLO-World)

<h1 id ="cont">
    <div align="center">Contribution</div>
</h1>

I'm excited to welcome new contributors! This project is still under development, and there's plenty of room for improvement. Whether you're a seasoned developer or just getting started, I value your contributions in all forms.


<h1 id="lic">
    <div align="center">License</div>
</h1>

This software is licensed under the **Affero General Public License v3.0 (AGPL-3.0)**. You can find the full license text here: [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.en.html).