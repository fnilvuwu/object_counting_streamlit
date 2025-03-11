from ultralytics import YOLO
import custom_counter
import cv2
import argparse
import time

parser = argparse.ArgumentParser(description='YOLOv8 Object Counting')

parser.add_argument('--model', type=str, required=True, help='Path to saved model')
parser.add_argument('--path', type=str, required=True, help='Path tp target video file')
parser.add_argument('--x_start', type=float, required=True, help='Fraction of the video width where the line starts (0.0 = left, 1.0 = right)')
parser.add_argument('--x_end', type=float, required=True, help='Fraction of the video width where the line ends (0.0 = left, 1.0 = right)')
parser.add_argument('--y_start', type=float, required=True, help='Fraction of the video height where the line starts (0.0 = bottom, 1.0 = top)')
parser.add_argument('--y_end', type=float, required=True, help='Fraction of the video height where the line ends (0.0 = bottom, 1.0 = top)')
parser.add_argument('--conf', type=float, default=0.2, help='Object confidence threshold (default: 0.2)')
parser.add_argument('--iou', type=float, default=0.3, help='IOU threshold for NMS (default: 0.3)')

args = parser.parse_args()

model_path = args.model # Path to saved YOLOv8 pytorch (.pt) model
video_path = args.path  # Path to saved video file

# Line position definition uses fractional system
# Where the x starts from the left side of the video frame width (0.0)
# And ends on the right side of the video frame width (1.0)

line_x_start = args.x_start
line_x_end = args.x_end

# The line y definition starts from the bottom side of the video frame height (0.0)
# And ends on the top side of the video frame height (1.0)

line_y_start = args.y_start
line_y_end = args.y_end

# Optional confidence & IoU threshold
conf_thresh = args.conf
iou_thresh = args.iou

# Initialize model
model = YOLO(model_path)

# Initialize graphics capture
cap = cv2.VideoCapture(video_path)

# Get video properties
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate line points based on provided fractional line definitions
line_points = [(int(args.x_start * w), int(args.y_start * h)), (int(args.x_end * w), int(args.y_end * h))]

# Initialize the custom object counter
counter = custom_counter.CustomCounter()
counter.set_args(view_img=False,  # Set to False for headless operation
                 reg_pts=line_points,
                 classes_names=model.names,
                 draw_tracks=True)

# Start timing
start_time = time.time()

print("Starting object counting..")

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        # print("Counting finished.")
        break

    # Perform object tracking and counting
    tracks = model.track(im0, persist=True, show=False, conf=conf_thresh, iou=iou_thresh, verbose=False, tracker='bytetrack.yaml')
    im0 = counter.start_counting(im0, tracks)

    # Optional: Save or process the frame (im0) as needed

# Time elapsed
end_time = time.time()
print(f"Object counting completed in {end_time - start_time:.2f} seconds.")
print("\n")

# Print final counts
print("In counts:", counter.in_counts)
print("Out counts:", counter.out_counts)
print("In Classes:", dict(counter.in_classes))
print("Out Classes:", dict(counter.out_classes))

cap.release()



