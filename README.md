# Fishcluster Vision-Based Fish Measurement Prototype

## Overview
This prototype demonstrates a computer vision pipeline for fish detection and size estimation in ponds.  
It is designed for **low-cost hardware** and reflects **field realism** in African aquaculture.

Farmers currently feed fish daily without knowing:
- Actual fish size distribution
- Growth rate vs. feed given
- Biomass in pond
- Whether feed is wasted or under-fed

This prototype provides visibility into the pond using simple cameras.

---

## Features
- Detect fish in video frames (YOLOv5s pretrained)
- Estimate relative fish length (bounding box → cm proxy)
- Approximate fish weight using fisheries formula
- Output metrics:
  - Average fish size & weight
  - Size distribution (small / medium / large)
  - Total biomass
  - Average detection confidence
- Annotated video with bounding boxes + size/weight overlays

---

## Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/fishcluster-trial.git
cd fishcluster-trial
pip install -r requirements.txt

Requirements
Python 3.8+

PyTorch

Torchvision

OpenCV

NumPy

Pillow

Matplotlib

tqdm

Ultralytics YOLOv5

Usage
Run detection on a sample video:

bash
python detect.py --input "sample_fish_video.mp4" --output runs/detect
Outputs
runs/detect/fish_detected.mp4 → annotated video

Console summary:

text
📊 Summary: Detected 38 fish
   Average length: 14.7 cm
   Average weight: 420.5 g
   Total biomass: 15979.0 g
   Average confidence: 0.41
📊 Size distribution:
   Small (<15 cm): 12
   Medium (15–30 cm): 23
   Large (>30 cm): 3
Assumptions
Camera is static above or beside pond.

Scale factor = 0.01 cm/pixel (approximate).

Constants (a=0.012, b=3.0) tuned for tilapia/catfish.

Bounding boxes may merge overlapping fish → inflated weights.

Limitations
Murky water, occlusion, reflections reduce accuracy.

Length estimates approximate without calibration.

Pretrained YOLO not specialized for fish.

Path to Improvement
Calibrate pixel → cm using reference object.

Retrain YOLO on fish dataset for tighter bounding boxes.

Deploy on GPU/edge devices for real-time fluency.

Integrate with feeder logic, harvest prediction, and financing.

Deliverables
detect.py (code)

Annotated video output

Console metrics (size, weight, biomass, confidence, distribution)

Technical note (Technical_Note.pdf)

Demo video (2–5 mins walkthrough)

Author
Elewa Prince Chizi  
Computer Vision / AI Engineer (Trial Candidate)
Fishcluster Paid Trial Task
