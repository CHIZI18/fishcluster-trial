import cv2
import torch
import argparse
import os

# --- Length-Weight Constants (generic tilapia/catfish) ---
a = 0.012  # constant
b = 3.0    # exponent

def estimate_length(bbox, scale_factor=0.01):
    """
    Estimate fish length from bounding box.
    bbox = (x_min, y_min, x_max, y_max)
    scale_factor = conversion from pixels to cm (assumption)
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    length_pixels = max(width, height)
    length_cm = length_pixels * scale_factor
    return length_cm

def estimate_weight(length_cm):
    """
    Estimate fish weight using length-weight formula W = a * L^b
    """
    return a * (length_cm ** b)

def classify_size(length_cm):
    """Classify fish size into Small, Medium, Large categories."""
    if length_cm < 15:
        return "Small"
    elif length_cm < 30:
        return "Medium"
    else:
        return "Large"

def annotate_frame(frame, detections):
    """
    Overlay size and weight estimates on detection frame.
    Returns: annotated frame, list of lengths, list of weights, list of sizes, list of confidences
    """
    lengths = []
    weights = []
    sizes = []
    confidences = []

    for det in detections:
        x_min, y_min, x_max, y_max, conf = det[:5]
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        length_cm = estimate_length(bbox, scale_factor=0.01)
        weight_g = estimate_weight(length_cm) * 1000  # convert to grams

        # Skip unrealistic bounding boxes
        if (bbox[2] - bbox[0]) > 300:
            continue

        # Cap max weight at 2kg
        if weight_g > 2000:
            weight_g = 2000

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)

        # Add text overlay
        text = f"L: {length_cm:.1f} cm, W: {weight_g:.1f} g"
        cv2.putText(frame, text, (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Store values for summary
        lengths.append(length_cm)
        weights.append(weight_g)
        sizes.append(classify_size(length_cm))
        confidences.append(conf)

    return frame, lengths, weights, sizes, confidences

def run_detection(input_path, output_path):
    # Load YOLOv5s model from Ultralytics
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    model.conf = 0.35  # confidence threshold (tune between 0.3–0.5)

    # Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {input_path}")
        return

    # Prepare video writer
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))  # ensure integer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(output_path, exist_ok=True)
    out_video = cv2.VideoWriter(os.path.join(output_path, 'fish_detected.mp4'),
                                fourcc, fps, (width, height))

    frame_count = 0
    all_lengths = []
    all_weights = []
    all_sizes = []
    all_confidences = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Optional: skip frames for fluency
        # if frame_count % 2 != 0:
        #     continue

        # Run YOLO detection
        results = model(frame)

        # Extract bounding boxes + confidence
        detections = results.xyxy[0].cpu().numpy()

        # Annotate frame with size + weight
        annotated_frame, lengths, weights, sizes, confidences = annotate_frame(frame, detections)
        all_lengths.extend(lengths)
        all_weights.extend(weights)
        all_sizes.extend(sizes)
        all_confidences.extend(confidences)

        # Write frame to output video
        out_video.write(annotated_frame)

        # Optional: show live preview window
        cv2.imshow("YOLO Fish Detection", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"✅ Detection complete. Annotated video saved to {output_path}/fish_detected.mp4")

    # Print summary metrics
    if all_lengths and all_weights:
        avg_length = sum(all_lengths) / len(all_lengths)
        avg_weight = sum(all_weights) / len(all_weights)
        avg_conf = sum(all_confidences) / len(all_confidences)
        total_biomass = sum(all_weights)

        print(f"📊 Summary: Detected {len(all_lengths)} fish")
        print(f"   Average length: {avg_length:.1f} cm")
        print(f"   Average weight: {avg_weight:.1f} g")
        print(f"   Total biomass: {total_biomass:.1f} g")
        print(f"   Average confidence: {avg_conf:.2f}")

        # Print size distribution
        small_count = all_sizes.count("Small")
        medium_count = all_sizes.count("Medium")
        large_count = all_sizes.count("Large")
        print("📊 Size distribution:")
        print(f"   Small (<15 cm): {small_count}")
        print(f"   Medium (15–30 cm): {medium_count}")
        print(f"   Large (>30 cm): {large_count}")
    else:
        print("⚠️ No fish detected for summary metrics.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder")
    args = parser.parse_args()

    run_detection(args.input, args.output)
