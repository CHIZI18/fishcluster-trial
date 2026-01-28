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

## Assumptions
Camera is static above or beside pond.
Scale factor = 0.01 cm/pixel (approximate).
Constants (a=0.012, b=3.0) tuned for tilapia/catfish.
Bounding boxes may merge overlapping fish → inflated weights.

## Limitations
Murky water, occlusion, reflections reduce accuracy.
Length estimates approximate without calibration.
Pretrained YOLO not specialized for fish.

## Path to Improvement
- Calibrate pixel → cm using reference objects
- Retrain YOLO on fish datasets for tighter bounding boxes
- Deploy on GPU/edge devices for real-time monitoring
- Integrate with feeder logic and harvest prediction

## Deliverables
- `detect.py` (code)
- Annotated video output: [fish_detected.mp4](https://drive.google.com/file/d/1liQKN4J0SWNi6xg5uwPT8VoTNMUbZP-4/view?usp=drive_link)
- Console metrics (size, weight, biomass, confidence, distribution)
- Technical Note: [Technical_Note.pdf](Technical_Note.pdf)
- Demo video (2–5 mins walkthrough): [Watch here](https://drive.google.com/file/d/1i3eBetShwvm4fajYy6PQ5JWgMo0sxmkF/view?usp=drive_link)

## Next Phase Plan
See [Next_Phase_Plan.pdf](Next_Phase_Plan.pdf) for the full roadmap.  
Key improvements include:
1. **Improved Training (More Epochs)** — retrain YOLOv5s with 100–300 epochs and fish‑specific datasets.  
2. **Nighttime Monitoring** — integrate infrared cameras and illumination for 24/7 tracking.  
3. **Infrared Light for Fish Tracking** — IR is invisible to fish but usable by cameras; retrain on IR footage.  
4. **Integration Thinking** — feeder logic, harvest prediction, financing support.  
5. **Next Deliverables** — retrained weights, IR demo footage, updated Technical Note.

## Author
Elewa Prince Chizi  
Computer Vision / AI Engineer (Trial Candidate)  
Fishcluster Paid Trial Task
