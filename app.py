import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import custom_counter
import os
import time
import sqlite3
import datetime
import pandas as pd
from PIL import Image, ImageDraw

# This will store our object tracking memory
track_memory = {}

# Set page config
st.set_page_config(layout="wide", page_title="YOLOv8 Object Detection & Counting")

# Initialize session state
if "line_start" not in st.session_state:
    st.session_state.line_start = None
    st.session_state.line_end = None
    st.session_state.processed_video = None
    st.session_state.processing_complete = False
    st.session_state.counter_results = None
    st.session_state.current_tab = "Video"  # Add a state to track current tab
    st.session_state.use_line = True  # Add state for line toggle

# Streamlit UI
st.title("YOLOv8 Object Detection & Counting")

# Create tabs for Video and Image processing
tab1, tab2, tab3 = st.tabs(
    ["Video Processing", "Image Processing", "Detection History"]
)


# Create a function to initialize the SQLite database
def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    conn = sqlite3.connect("detection_history.db")
    c = conn.cursor()

    # Create table for video detections
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS video_detections (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        video_name TEXT,
        frame_number INTEGER,
        track_id INTEGER,
        class_id INTEGER,
        class_name TEXT,
        confidence REAL,
        x1 INTEGER,
        y1 INTEGER,
        x2 INTEGER,
        y2 INTEGER,
        counted BOOLEAN,
        direction TEXT
    )
    """
    )

    # Create table for image detections
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS image_detections (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        image_name TEXT,
        class_id INTEGER,
        class_name TEXT,
        confidence REAL,
        x1 INTEGER,
        y1 INTEGER,
        x2 INTEGER,
        y2 INTEGER
    )
    """
    )

    # Create table for detection sessions
    c.execute(
        """
    CREATE TABLE IF NOT EXISTS detection_sessions (
        id INTEGER PRIMARY KEY,
        timestamp TEXT,
        file_name TEXT,
        file_type TEXT,
        total_detections INTEGER,
        settings TEXT
    )
    """
    )

    conn.commit()
    conn.close()


# Initialize database when the app starts
init_database()


# Function to save detection session
def save_detection_session(file_name, file_type, total_detections, settings):
    conn = sqlite3.connect("detection_history.db")
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        """
    INSERT INTO detection_sessions (timestamp, file_name, file_type, total_detections, settings)
    VALUES (?, ?, ?, ?, ?)
    """,
        (timestamp, file_name, file_type, total_detections, settings),
    )

    session_id = c.lastrowid
    conn.commit()
    conn.close()
    return session_id


# Function to save video detection
def save_video_detection(
    video_name,
    frame_number,
    track_id,
    class_id,
    class_name,
    confidence,
    x1,
    y1,
    x2,
    y2,
    counted,
    direction,
):
    conn = sqlite3.connect("detection_history.db")
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        """
    INSERT INTO video_detections 
    (timestamp, video_name, frame_number, track_id, class_id, class_name, confidence, 
     x1, y1, x2, y2, counted, direction)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            timestamp,
            video_name,
            frame_number,
            track_id,
            class_id,
            class_name,
            confidence,
            x1,
            y1,
            x2,
            y2,
            counted,
            direction,
        ),
    )

    conn.commit()
    conn.close()


# Function to save image detection
def save_image_detection(image_name, class_id, class_name, confidence, x1, y1, x2, y2):
    conn = sqlite3.connect("detection_history.db")
    c = conn.cursor()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute(
        """
    INSERT INTO image_detections 
    (timestamp, image_name, class_id, class_name, confidence, x1, y1, x2, y2)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (timestamp, image_name, class_id, class_name, confidence, x1, y1, x2, y2),
    )

    conn.commit()
    conn.close()


# Function to reset sidebar based on active tab
def update_sidebar_for_tab(tab_name):
    # Clear the sidebar first
    st.sidebar.empty()

    # Set sidebar content based on active tab
    st.sidebar.header(f"{tab_name} Settings")

    if tab_name == "Video":
        # Video processing sidebar
        uploaded_file = st.sidebar.file_uploader(
            "Choose a video...", type=["mp4", "avi", "mov"], key="video_uploader"
        )

        # Adjustable confidence and IoU thresholds
        conf_thresh = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.2, 0.05, key="video_conf"
        )
        iou_thresh = st.sidebar.slider(
            "IoU Threshold", 0.0, 1.0, 0.3, 0.05, key="video_iou"
        )

        # Add option to use counting line
        use_line = st.sidebar.checkbox(
            "Use Counting Line", value=st.session_state.use_line, key="use_line_toggle"
        )

        # Update session state
        if use_line != st.session_state.use_line:
            st.session_state.use_line = use_line

        return uploaded_file, conf_thresh, iou_thresh, None, use_line

    else:  # Image tab
        # Image processing sidebar
        uploaded_image = st.sidebar.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_uploader",
        )

        # Adjustable confidence and IoU thresholds
        img_conf_thresh = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="img_conf"
        )
        img_iou_thresh = st.sidebar.slider(
            "IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="img_iou"
        )

        # Option to show labels and confidence scores
        show_labels = st.sidebar.checkbox("Show Labels", value=True)
        show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)

        return (
            uploaded_image,
            img_conf_thresh,
            img_iou_thresh,
            (show_labels, show_conf),
            None,
        )


# Add this new function to the app.py file, near your other processing functions


def process_total_counting(
    frame, tracks, class_names, counter, class_counts, track_memory
):
    """
    Process frame for total object counting without duplicates by using track memory

    Args:
        frame: The current video frame
        tracks: Detection tracks from YOLO
        class_names: Dictionary of class names
        counter: Counter object to store counts
        class_counts: Dictionary to track counts by class
        track_memory: Dictionary to track objects across frames and prevent duplicates
    """
    # Create a copy of the frame
    annotated_frame = frame.copy()
    frame_height, frame_width = frame.shape[:2]

    # Get current frame's track IDs
    current_tracks = set()
    new_tracks = set()

    # Ensure counter has total_counts attribute
    if not hasattr(counter, "total_counts"):
        counter.total_counts = 0

    # Get all detected objects
    if not tracks or len(tracks) == 0:
        return annotated_frame, track_memory

    boxes = tracks[0].boxes
    if boxes is None or len(boxes) == 0:
        return annotated_frame, track_memory

    # Track current frame's objects and update memory
    for box in boxes:
        # Get box coordinates and convert to integers
        if hasattr(box, "xyxy"):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
        else:
            continue

        # Calculate box center and area
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_area = (x2 - x1) * (y2 - y1)

        # Get class ID and convert to integer
        if hasattr(box, "cls"):
            cls_id = int(box.cls[0])
        else:
            cls_id = 0

        # Skip if class is not in allowed classes
        if (
            hasattr(counter, "allowed_classes")
            and cls_id not in counter.allowed_classes
        ):
            continue

        # Get track ID if available
        if hasattr(box, "id") and box.id is not None:
            track_id = int(box.id[0])
            current_tracks.add(track_id)

            # Check if this is a new track or existing track
            if track_id not in track_memory:
                # New track found - add to memory
                track_memory[track_id] = {
                    "class_id": cls_id,
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "positions": [(center_x, center_y)],
                    "counted": False,
                    "frames_visible": 1,
                    "area": box_area,
                }
                new_tracks.add(track_id)
            else:
                # Existing track - update last seen and positions
                track_memory[track_id]["last_seen"] = time.time()
                track_memory[track_id]["positions"].append((center_x, center_y))
                track_memory[track_id]["frames_visible"] += 1
                track_memory[track_id]["area"] = box_area  # Update area

                # Keep only the last 30 positions to save memory
                if len(track_memory[track_id]["positions"]) > 30:
                    track_memory[track_id]["positions"].pop(0)
        else:
            continue

        # Draw bounding box
        confidence_color = (0, 255, 0)  # Green for counted objects
        if track_id in track_memory and not track_memory[track_id]["counted"]:
            # Only count an object when it's been tracked for at least 5 frames
            # and is not at the edge of the frame (to avoid counting partial objects)
            edge_margin = 20  # pixels from edge
            centered_in_frame = edge_margin < center_x < (
                frame_width - edge_margin
            ) and edge_margin < center_y < (frame_height - edge_margin)

            if track_memory[track_id]["frames_visible"] >= 5 and centered_in_frame:
                # Mark as counted
                track_memory[track_id]["counted"] = True

                # Update class counts
                if cls_id in class_counts:
                    class_counts[cls_id] += 1
                else:
                    class_counts[cls_id] = 1

                confidence_color = (0, 255, 0)  # Green for counted objects
            else:
                confidence_color = (
                    0,
                    165,
                    255,
                )  # Orange for tracking but not yet counted
        elif track_id not in track_memory:
            confidence_color = (0, 0, 255)  # Red for new detections

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), confidence_color, 2)

        # Get class name
        if isinstance(class_names, dict):
            cls_name = class_names.get(cls_id, f"Class {cls_id}")
        else:
            cls_name = f"Class {cls_id}"

        # Draw label with track ID
        if track_id in track_memory and track_memory[track_id]["counted"]:
            label = f"{cls_name}-{track_id} ✓"  # Checkmark for counted objects
        else:
            label = f"{cls_name}-{track_id}"

        cv2.putText(
            annotated_frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            confidence_color,
            2,
        )

    # Clean up tracks that haven't been seen recently (5 seconds timeout)
    current_time = time.time()
    tracks_to_remove = []

    for track_id, track_data in track_memory.items():
        if current_time - track_data["last_seen"] > 5.0:  # 5 second timeout
            tracks_to_remove.append(track_id)

    for track_id in tracks_to_remove:
        del track_memory[track_id]

    # Update total count based on counted objects
    counter.total_counts = sum(class_counts.values())

    # Add count and indicators to the frame
    cv2.putText(
        annotated_frame,
        f"Total Count: {counter.total_counts}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    # Add tracking stats
    cv2.putText(
        annotated_frame,
        f"Tracking: {len(current_tracks)} objects ({len(new_tracks)} new)",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2,
    )

    return annotated_frame, track_memory


# Video Processing Tab
with tab1:
    st.session_state.current_tab = "Video"

    # Get video sidebar inputs
    uploaded_file, conf_thresh, iou_thresh, _, use_line = update_sidebar_for_tab(
        "Video"
    )

    st.header("Video Object Counting")

    if uploaded_file:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()

        # Display original video
        st.video(video_path)

        # Extract first frame for line drawing (if line is enabled)
        cap = cv2.VideoCapture(video_path)
        success, first_frame = cap.read()
        cap.release()

        if success:
            # Store video dimensions
            frame_height, frame_width = first_frame.shape[:2]

            # Convert to RGB for display
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(first_frame_rgb)

            if use_line:
                # Initialize line coordinates if not set
                if st.session_state.line_start is None:
                    st.session_state.line_start = (
                        int(frame_width * 0.2),
                        int(frame_height * 0.5),
                    )
                    st.session_state.line_end = (
                        int(frame_width * 0.8),
                        int(frame_height * 0.5),
                    )

                # Create columns for the UI layout
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Set Counting Line")

                    # Create a copy of the image with the line drawn on it
                    image_with_line = pil_image.copy()
                    draw = ImageDraw.Draw(image_with_line)

                    # Draw the line
                    draw.line(
                        [st.session_state.line_start, st.session_state.line_end],
                        fill=(255, 0, 0),
                        width=5,
                    )

                    # Draw the start point (green)
                    draw.ellipse(
                        [
                            (
                                st.session_state.line_start[0] - 5,
                                st.session_state.line_start[1] - 5,
                            ),
                            (
                                st.session_state.line_start[0] + 5,
                                st.session_state.line_start[1] + 5,
                            ),
                        ],
                        fill=(0, 255, 0),
                    )

                    # Draw the end point (blue)
                    draw.ellipse(
                        [
                            (
                                st.session_state.line_end[0] - 5,
                                st.session_state.line_end[1] - 5,
                            ),
                            (
                                st.session_state.line_end[0] + 5,
                                st.session_state.line_end[1] + 5,
                            ),
                        ],
                        fill=(0, 0, 255),
                    )

                    # Display the image with line
                    st.image(
                        image_with_line,
                        caption="Counting Line",
                        use_container_width=True,
                    )

                    # Manual coordinate inputs with sliders for easier adjustment
                    st.subheader("Adjust Line Position")

                    # Create two rows of sliders
                    col_start_x, col_start_y = st.columns(2)
                    col_end_x, col_end_y = st.columns(2)

                    # Adjust start point
                    new_start_x = col_start_x.slider(
                        "Start X",
                        0,
                        frame_width,
                        st.session_state.line_start[0],
                        key="start_x_slider",
                    )
                    new_start_y = col_start_y.slider(
                        "Start Y",
                        0,
                        frame_height,
                        st.session_state.line_start[1],
                        key="start_y_slider",
                    )

                    # Adjust end point
                    new_end_x = col_end_x.slider(
                        "End X",
                        0,
                        frame_width,
                        st.session_state.line_end[0],
                        key="end_x_slider",
                    )
                    new_end_y = col_end_y.slider(
                        "End Y",
                        0,
                        frame_height,
                        st.session_state.line_end[1],
                        key="end_y_slider",
                    )

                    # Update line coordinates if changed
                    if (
                        new_start_x != st.session_state.line_start[0]
                        or new_start_y != st.session_state.line_start[1]
                        or new_end_x != st.session_state.line_end[0]
                        or new_end_y != st.session_state.line_end[1]
                    ):

                        st.session_state.line_start = (
                            int(new_start_x),
                            int(new_start_y),
                        )
                        st.session_state.line_end = (int(new_end_x), int(new_end_y))
                        st.rerun()

                with col2:
                    st.subheader("Processing Options")

                    # Display current line coordinates
                    st.write("Current Line Coordinates:")
                    st.write(f"Start: {st.session_state.line_start}")
                    st.write(f"End: {st.session_state.line_end}")

                    # Set line points for counting
                    line_points = [
                        st.session_state.line_start,
                        st.session_state.line_end,
                    ]

                    # Process video button
                    process_button = st.button(
                        "Process Video", type="primary", use_container_width=True
                    )
            else:
                # Simplified UI when line is disabled
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Preview Frame")
                    # Show first frame without line
                    st.image(pil_image, caption="First Frame", use_container_width=True)

                with col2:
                    st.subheader("Processing Options")
                    st.info(
                        "Counting Line is disabled. All objects will be counted without directional distinction."
                    )

                    # Process video button
                    process_button = st.button(
                        "Process Video", type="primary", use_container_width=True
                    )

                # Empty line points when line is disabled
                line_points = []

            # In the video processing section where you're processing the button click
            if process_button:
                # Load YOLO model
                with st.spinner("Loading YOLO model..."):
                    model = YOLO("yolo11s.pt")  # Change this to your trained model path

                # Define allowed classes
                allowed_classes = [
                    1,
                    2,
                    3,
                    5,
                    7,
                ]  # bicycle, car, motorcycle, bus, truck

                # Open video
                cap = cv2.VideoCapture(video_path)
                w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                )
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Setup counter with or without line
                counter = custom_counter.CustomCounter()
                counter.set_args(
                    view_img=False,
                    reg_pts=line_points if use_line else [],
                    classes_names=model.names,
                    draw_tracks=True,
                    use_line=use_line,  # Pass the line toggle to the counter
                    allowed_classes=allowed_classes,  # Pass the allowed classes to the counter
                )

                # Create a temporary directory for output
                temp_output_dir = tempfile.mkdtemp()
                out_path = os.path.join(temp_output_dir, "processed_output.mp4")

                # Progress indicators
                progress_container = st.container()

                with progress_container:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    progress_text.text("Processing video frames...")

                # Use H.264 codec for better compatibility
                fourcc = (
                    cv2.VideoWriter_fourcc(*"H264")
                    if cv2.VideoWriter_fourcc(*"H264") != -1
                    else cv2.VideoWriter_fourcc(*"mp4v")
                )
                out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                # Track total object count when not using line
                total_objects = 0
                class_counts = {}
                frame_idx = 0
                total_detections = 0
                video_name = os.path.basename(uploaded_file.name)

                # Save detection session
                settings_info = (
                    f"conf={conf_thresh}, iou={iou_thresh}, use_line={use_line}"
                )
                session_id = save_detection_session(
                    video_name, "video", 0, settings_info
                )

                # Reset video capture to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                # Process the video in a single loop
                while cap.isOpened():
                    success, im0 = cap.read()
                    if not success:
                        break

                    # Run tracking with allowed classes filter
                    tracks = model.track(
                        im0,
                        persist=True,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False,
                        tracker="custom_tracker.yaml",
                        classes=allowed_classes,  # Filter classes during detection
                    )

                    # Process frame based on counting method
                    if use_line:
                        # Use the existing line counting functionality
                        im0 = counter.start_counting(im0, tracks)

                        # Save detections to database
                        if tracks and len(tracks) > 0 and hasattr(tracks[0], "boxes"):
                            boxes = tracks[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                for i, box in enumerate(boxes):
                                    if (
                                        hasattr(box, "cls")
                                        and hasattr(box, "conf")
                                        and hasattr(box, "xyxy")
                                        and hasattr(box, "id")
                                    ):
                                        cls_id = int(box.cls[0])

                                        # Skip classes not in allowed list
                                        if cls_id not in allowed_classes:
                                            continue

                                        conf = float(box.conf[0])
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        track_id = (
                                            int(box.id[0]) if box.id is not None else -1
                                        )
                                        cls_name = model.names[cls_id]

                                        # Determine if this object was counted by checking counter's counting_list
                                        counted = track_id in counter.counting_list

                                        # Determine direction
                                        direction = "none"
                                        if counted:
                                            # Check if this track_id is in in_classes or out_classes
                                            if track_id in counter.in_classes:
                                                direction = "in"
                                            elif track_id in counter.out_classes:
                                                direction = "out"

                                        # Save detection to database
                                        save_video_detection(
                                            video_name,
                                            frame_idx,
                                            track_id,
                                            cls_id,
                                            cls_name,
                                            conf,
                                            x1,
                                            y1,
                                            x2,
                                            y2,
                                            counted,
                                            direction,
                                        )
                                        total_detections += 1
                    else:
                        # Use the improved counting function that prevents duplicates
                        im0, track_memory = process_total_counting(
                            im0,
                            tracks,
                            model.names,
                            counter,
                            class_counts,
                            track_memory,
                        )

                        # Save detections to database
                        if tracks and len(tracks) > 0 and hasattr(tracks[0], "boxes"):
                            boxes = tracks[0].boxes
                            if boxes is not None and len(boxes) > 0:
                                for i, box in enumerate(boxes):
                                    if (
                                        hasattr(box, "cls")
                                        and hasattr(box, "conf")
                                        and hasattr(box, "xyxy")
                                        and hasattr(box, "id")
                                    ):
                                        cls_id = int(box.cls[0])

                                        # Skip classes not in allowed list
                                        if cls_id not in allowed_classes:
                                            continue

                                        conf = float(box.conf[0])
                                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                                        track_id = (
                                            int(box.id[0]) if box.id is not None else -1
                                        )
                                        cls_name = model.names[cls_id]

                                        # Check if this track is counted in track_memory
                                        counted = False
                                        if (
                                            track_id in track_memory
                                            and track_memory[track_id]["counted"]
                                        ):
                                            counted = True

                                        # Save detection to database
                                        save_video_detection(
                                            video_name,
                                            frame_idx,
                                            track_id,
                                            cls_id,
                                            cls_name,
                                            conf,
                                            x1,
                                            y1,
                                            x2,
                                            y2,
                                            counted,
                                            "total",
                                        )
                                        total_detections += 1

                    # Write processed frame to output video
                    out.write(im0)

                    # Update progress
                    frame_idx += 1
                    progress_bar.progress(frame_idx / frame_count)

                # Release resources
                cap.release()
                out.release()

                # Update session with final detection count
                conn = sqlite3.connect("detection_history.db")
                c = conn.cursor()
                c.execute(
                    "UPDATE detection_sessions SET total_detections = ? WHERE id = ?",
                    (total_detections, session_id),
                )
                conn.commit()
                conn.close()
                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                progress_container.success("Processing completed!")

                # Store results in session state
                st.session_state.processed_video = out_path
                st.session_state.processing_complete = True

                if use_line:
                    # Store directional results when using line
                    st.session_state.counter_results = {
                        "in_counts": counter.in_counts,
                        "out_counts": counter.out_counts,
                        "in_classes": counter.in_classes,
                        "out_classes": counter.out_classes,
                        "model_names": model.names,
                        "use_line": True,
                    }
                else:
                    # Store total count results when not using line
                    st.session_state.counter_results = {
                        "total_counts": (
                            counter.total_counts
                            if hasattr(counter, "total_counts")
                            else sum(class_counts.values())
                        ),
                        "class_counts": class_counts,
                        "model_names": model.names,
                        "use_line": False,
                    }

                # Force page refresh to show results
                st.rerun()

            # Display results section after processing is complete
            if (
                st.session_state.processing_complete
                and st.session_state.processed_video
            ):
                st.header("Processing Results")

                # Check if line was used for this result
                if st.session_state.counter_results.get("use_line", True):
                    # Display directional counting metrics
                    st.subheader("Directional Counting Results")
                    col_in, col_out = st.columns(2)
                    col_in.metric(
                        "Objects In", st.session_state.counter_results["in_counts"]
                    )
                    col_out.metric(
                        "Objects Out", st.session_state.counter_results["out_counts"]
                    )

                    # Display class details in an expandable section
                    with st.expander("Detailed Class Counts", expanded=True):
                        col_in_detail, col_out_detail = st.columns(2)

                        with col_in_detail:
                            st.write("**In Classes:**")
                            for cls, count in st.session_state.counter_results[
                                "in_classes"
                            ].items():
                                # Fixed: No conversion attempt, using cls directly as the key or displaying it as is
                                if isinstance(
                                    st.session_state.counter_results["model_names"],
                                    dict,
                                ):
                                    # If model_names is a dictionary, try to get the class name
                                    # Handle both integer and string keys
                                    try:
                                        # Try with the original key
                                        cls_name = st.session_state.counter_results[
                                            "model_names"
                                        ].get(cls, cls)
                                        # If not found and cls is a string that might represent a number
                                        if (
                                            cls_name == cls
                                            and isinstance(cls, str)
                                            and cls.isdigit()
                                        ):
                                            cls_name = st.session_state.counter_results[
                                                "model_names"
                                            ].get(int(cls), cls)
                                    except (ValueError, TypeError):
                                        cls_name = cls
                                else:
                                    # If model_names is not a dictionary, just use the class key as is
                                    cls_name = cls
                                st.write(f"- {cls_name}: {count}")

                        with col_out_detail:
                            st.write("**Out Classes:**")
                            for cls, count in st.session_state.counter_results[
                                "out_classes"
                            ].items():
                                # Fixed: Same approach as above
                                if isinstance(
                                    st.session_state.counter_results["model_names"],
                                    dict,
                                ):
                                    try:
                                        cls_name = st.session_state.counter_results[
                                            "model_names"
                                        ].get(cls, cls)
                                        if (
                                            cls_name == cls
                                            and isinstance(cls, str)
                                            and cls.isdigit()
                                        ):
                                            cls_name = st.session_state.counter_results[
                                                "model_names"
                                            ].get(int(cls), cls)
                                    except (ValueError, TypeError):
                                        cls_name = cls
                                else:
                                    cls_name = cls
                                st.write(f"- {cls_name}: {count}")
                else:
                    # Display total counting metrics
                    st.subheader("Total Object Counts")
                    st.metric(
                        "Total Objects Detected",
                        st.session_state.counter_results["total_counts"],
                    )

                    # Display class details
                    with st.expander("Detailed Class Counts", expanded=True):
                        class_counts = st.session_state.counter_results["class_counts"]
                        model_names = st.session_state.counter_results["model_names"]

                        for cls, count in class_counts.items():
                            if isinstance(model_names, dict):
                                try:
                                    cls_name = model_names.get(cls, cls)
                                    if (
                                        cls_name == cls
                                        and isinstance(cls, str)
                                        and cls.isdigit()
                                    ):
                                        cls_name = model_names.get(int(cls), cls)
                                except (ValueError, TypeError):
                                    cls_name = cls
                            else:
                                cls_name = cls
                            st.write(f"- {cls_name}: {count}")

                # Display the processed video
                st.subheader("Processed Video")
                st.video(st.session_state.processed_video)

                # Offer download button
                with open(st.session_state.processed_video, "rb") as file:
                    st.download_button(
                        label="Download Processed Video",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )

    else:
        # Display instruction when no file is uploaded
        st.info("Please upload a video file to begin.")
        st.markdown(
            """
        ### How to use this app:
        1. Upload a video file using the sidebar
        2. Choose whether to use a counting line:
           - With line: Count objects crossing the line in both directions
           - Without line: Count all objects in the video
        3. If using a line, adjust the counting line position
        4. Click 'Process Video' to start object counting
        5. View results and download the processed video
        """
        )

# Image Processing Tab
with tab2:
    st.session_state.current_tab = "Image"

    # Get image sidebar inputs
    uploaded_image, img_conf_thresh, img_iou_thresh, vis_options, _ = (
        update_sidebar_for_tab("Image")
    )

    if vis_options:
        show_labels, show_conf = vis_options

    st.header("Image Object Detection")

    if uploaded_image:
        # Save uploaded image temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.read())
        image_path = tfile.name
        tfile.close()

        # Display original image
        img = Image.open(image_path)
        st.subheader("Original Image")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Process image button
        process_image_button = st.button(
            "Detect Objects", type="primary", use_container_width=True, key="detect_btn"
        )

        if process_image_button:
            # Load YOLO model
            with st.spinner("Loading YOLO model and processing image..."):
                model = YOLO("yolo11s.pt")  # Change this to your trained model path

                # Run detection
                results = model(
                    image_path,
                    conf=img_conf_thresh,
                    iou=img_iou_thresh,
                    verbose=False,
                    classes=[1, 2, 3, 5, 7],
                )

                # Convert cv2 image format for visualization
                img_cv = cv2.imread(image_path)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # Process and visualize results
                processed_image = img_cv.copy()

                # Get detection results
                boxes = results[0].boxes
                class_counts = {}
                total_detections = 0
                image_name = os.path.basename(uploaded_image.name)

                # Save detection session
                settings_info = f"conf={img_conf_thresh}, iou={img_iou_thresh}"
                session_id = save_detection_session(
                    image_name, "image", 0, settings_info
                )

                # Create a temporary directory for output
                temp_output_dir = tempfile.mkdtemp()
                result_path = os.path.join(temp_output_dir, "detected_image.jpg")

                # Draw bounding boxes on the image
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name
                    cls_name = model.names[cls_id]

                    # Count detections by class
                    if cls_name in class_counts:
                        class_counts[cls_name] += 1
                    else:
                        class_counts[cls_name] = 1

                    # Save detection to database
                    save_image_detection(
                        image_name, cls_id, cls_name, conf, x1, y1, x2, y2
                    )
                    total_detections += 1

                    # Draw bounding box
                    color = (255, 0, 0)  # Red for all boxes
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)

                    # Add label if enabled
                    if show_labels:
                        label = f"{cls_name}"
                        if show_conf:
                            label += f" {conf:.2f}"

                        # Calculate text size
                        text_size = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )[0]

                        # Fill background for text
                        cv2.rectangle(
                            processed_image,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color,
                            -1,
                        )

                        # Add text
                        cv2.putText(
                            processed_image,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

                # Update session with final detection count
                conn = sqlite3.connect("detection_history.db")
                c = conn.cursor()
                c.execute(
                    "UPDATE detection_sessions SET total_detections = ? WHERE id = ?",
                    (total_detections, session_id),
                )
                conn.commit()
                conn.close()

                # Save the processed image
                cv2.imwrite(
                    result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                )

                # Display results
                st.subheader("Detection Results")

                # Show object counts
                st.write("**Detected Objects:**")

                # Create a nice grid of metrics
                if class_counts:
                    cols = st.columns(min(3, len(class_counts)))
                    for i, (cls_name, count) in enumerate(class_counts.items()):
                        cols[i % len(cols)].metric(f"{cls_name}", f"{count}")
                else:
                    st.warning(
                        "No objects detected. Try adjusting the confidence threshold."
                    )

                # Show the processed image
                st.image(
                    processed_image,
                    caption="Detection Results",
                    use_container_width=True,
                )

                # Offer download button
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Image",
                        data=file,
                        file_name="detected_image.jpg",
                        mime="image/jpeg",
                        key="download_img",
                    )
    else:
        # Display instruction when no file is uploaded
        st.info("Please upload an image file to begin detection.")
        st.markdown(
            """
        ### How to use image detection:
        1. Upload an image file using the sidebar
        2. Adjust the confidence and IoU thresholds to control detection sensitivity
        3. Configure visualization options (labels, confidence scores)
        4. Click 'Detect Objects' to start detection
        5. View results and download the processed image
        """
        )

# In the History Tab section, update the clear history buttons to trigger page rerun
with tab3:
    st.session_state.current_tab = "History"

    st.header("Detection History")

    # Add a button to display all history at the top of the history tab
    if st.button("Refresh History Data", type="primary", key="refresh_history"):
        st.rerun()

    # Create subtabs for different history views
    history_tab1, history_tab2 = st.tabs(["Session History", "Detection Details"])

    with history_tab1:
        st.subheader("Processing Sessions")

        # Fetch session data from database
        conn = sqlite3.connect("detection_history.db")
        sessions_df = pd.read_sql_query(
            "SELECT * FROM detection_sessions ORDER BY timestamp DESC", conn
        )
        conn.close()

        if not sessions_df.empty:
            st.dataframe(sessions_df, use_container_width=True)

            # Allow downloading the session history as CSV
            csv = sessions_df.to_csv(index=False)
            st.download_button(
                label="Download Session History",
                data=csv,
                file_name="detection_sessions.csv",
                mime="text/csv",
            )
        else:
            st.info("No detection sessions found. Process some videos or images first.")

    with history_tab2:
        st.subheader("Detection Details")

        # Add filter options
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

        with filter_col1:
            file_type = st.radio("File Type", ["Video", "Image", "All"])

        with filter_col2:
            limit = st.slider("Number of records", 10, 5000, 100, 10)

        with filter_col3:
            st.write("&nbsp;")  # Add some spacing
            st.write("&nbsp;")  # Add some spacing
            show_all = st.checkbox("Show All Data", value=False)

        # Build query based on filters
        if file_type == "Video":
            if show_all:
                query = "SELECT * FROM video_detections ORDER BY timestamp DESC"
            else:
                query = f"SELECT * FROM video_detections ORDER BY timestamp DESC LIMIT {limit}"
            table_name = "video_detections"
        elif file_type == "Image":
            if show_all:
                query = "SELECT * FROM image_detections ORDER BY timestamp DESC"
            else:
                query = f"SELECT * FROM image_detections ORDER BY timestamp DESC LIMIT {limit}"
            table_name = "image_detections"
        else:
            # For "All", we need to handle differently since tables have different columns
            st.warning("Please select either Video or Image to view detection details.")
            query = None
            table_name = None

        if query and table_name:
            conn = sqlite3.connect("detection_history.db")
            detections_df = pd.read_sql_query(query, conn)
            conn.close()

            if not detections_df.empty:
                st.dataframe(detections_df, use_container_width=True)

                # Allow downloading the detection details as CSV
                csv = detections_df.to_csv(index=False)
                st.download_button(
                    label=f"Download {file_type} Detections",
                    data=csv,
                    file_name=f"{table_name}.csv",
                    mime="text/csv",
                )
            else:
                st.info(f"No {file_type.lower()} detections found.")

        # Add option to clear history
        with st.expander("Clear History"):
            st.warning("This action cannot be undone!")
            clear_col1, clear_col2, clear_col3 = st.columns(3)

            with clear_col1:
                if st.button("Clear Video Detections"):
                    conn = sqlite3.connect("detection_history.db")
                    c = conn.cursor()
                    c.execute("DELETE FROM video_detections")
                    conn.commit()
                    conn.close()
                    st.success("Video detection history cleared!")
                    # Add a rerun to refresh the page
                    time.sleep(1)  # Short delay for the success message to be visible
                    st.rerun()

            with clear_col2:
                if st.button("Clear Image Detections"):
                    conn = sqlite3.connect("detection_history.db")
                    c = conn.cursor()
                    c.execute("DELETE FROM image_detections")
                    conn.commit()
                    conn.close()
                    st.success("Image detection history cleared!")
                    # Add a rerun to refresh the page
                    time.sleep(1)  # Short delay for the success message to be visible
                    st.rerun()

            with clear_col3:
                if st.button(
                    "Clear All History", type="primary", use_container_width=True
                ):
                    conn = sqlite3.connect("detection_history.db")
                    c = conn.cursor()
                    c.execute("DELETE FROM video_detections")
                    c.execute("DELETE FROM image_detections")
                    c.execute("DELETE FROM detection_sessions")
                    conn.commit()
                    conn.close()
                    st.success("All detection history cleared!")
                    # Add a rerun to refresh the page
                    time.sleep(1)  # Short delay for the success message to be visible
                    st.rerun()
