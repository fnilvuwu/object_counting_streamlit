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
import atexit
import json

# Initialize session state variables if they don't exist
if "track_memory" not in st.session_state:
    st.session_state.track_memory = {}
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "processed_video" not in st.session_state:
    st.session_state.processed_video = None
if "counter_results" not in st.session_state:
    st.session_state.counter_results = None
if "line_start" not in st.session_state:
    st.session_state.line_start = None
if "line_end" not in st.session_state:
    st.session_state.line_end = None
if "video_dimensions" not in st.session_state:
    st.session_state.video_dimensions = None
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "Video"
if "use_line" not in st.session_state:
    st.session_state.use_line = True
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []

# This will store our object tracking memory
track_memory = st.session_state.track_memory

# Set page config
st.set_page_config(layout="wide", page_title="Penghitung Objek YOLOv8")

# Define allowed classes (1: bicycle, 2: car, 3: motorcycle, 5: bus, 7: truck)
allowed_classes = [1, 2, 3, 5, 7]

# Streamlit UI
st.title("Penghitung Objek YOLOv8")

# Create tabs
tab1, tab2, tab3 = st.tabs(
    ["Pemrosesan Video", "Pemrosesan Gambar", "Riwayat Deteksi"]
)

# Create a function to initialize the SQLite database
def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    conn = sqlite3.connect("detection_history.db")
    c = conn.cursor()

    # Check if the table already exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='detection_sessions'")
    table_exists = c.fetchone()
    
    # Only create the table if it doesn't exist
    if not table_exists:
        # Create table for detection sessions with final counts
        c.execute(
            """
        CREATE TABLE IF NOT EXISTS detection_sessions (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            file_name TEXT,
            file_type TEXT,
            total_detections INTEGER,
            class_counts TEXT,
            video_dimensions TEXT,
            settings_info TEXT
        )
        """
        )
        print("Database table created")
    else:
        print("Database table already exists")

    conn.commit()
    conn.close()

# Initialize database when the app starts
init_database()

# Function to save detection session to database
def save_detection_session(file_name, file_type, total_detections, class_counts, video_dimensions=None):
    """
    Save detection session details to SQLite database
    
    Args:
        file_name: Name of the processed file
        file_type: Type of file (video or image)
        total_detections: Total number of detections
        class_counts: Dictionary of class counts
        video_dimensions: Dimensions of the video (width, height)
    """
    try:
        conn = sqlite3.connect("detection_history.db")
        c = conn.cursor()
        
        # Translate file type to Indonesian
        if file_type.lower() == "video":
            file_type_id = "video"
        elif file_type.lower() == "image":
            file_type_id = "gambar"
        else:
            file_type_id = file_type.lower()
            
        # Convert class counts to JSON string
        class_counts_json = json.dumps(class_counts)
        
        # Convert video dimensions to string
        video_dimensions_str = str(video_dimensions) if video_dimensions else None
        
        # Create settings info string
        settings_info = f"File: {file_name}, Type: {file_type_id}, Dimensions: {video_dimensions_str}"
        
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert data into database
        c.execute(
            """
            INSERT INTO detection_sessions 
            (timestamp, file_name, file_type, total_detections, class_counts, video_dimensions, settings_info)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                file_name,
                file_type_id,
                total_detections,
                class_counts_json,
                video_dimensions_str,
                settings_info,
            ),
        )
        
        conn.commit()
        conn.close()
        print(f"Saved detection session to database: {file_name}, {total_detections} detections")
    except Exception as e:
        print(f"Error saving to database: {e}")

# Function to get video dimensions
def get_video_dimensions(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return (width, height)

# Function for total counting (non-line based)
def process_total_counting(im0, tracks, track_memory=None, conf_thresh=0.5):
    """
    Process frames for total object counting without using crossing lines.
    This function prevents duplicate counting by tracking object IDs.
    
    Args:
        im0: The input frame
        tracks: Detection tracks from YOLO
        track_memory: Dictionary to keep track of objects that have been counted
        conf_thresh: Confidence threshold for detections
        
    Returns:
        im0: Processed frame with annotation
        track_memory: Updated memory dictionary
    """
    if track_memory is None:
        track_memory = {}
    
    # Clone the image for drawing
    annotated_frame = im0.copy()
    
    try:
        if not tracks or len(tracks) == 0 or not hasattr(tracks[0], 'boxes') or tracks[0].boxes is None:
            return im0, track_memory
            
        boxes = tracks[0].boxes
        if len(boxes) == 0:
            return im0, track_memory
            
        # Process detections
        for i, box in enumerate(boxes):
            # Skip if no confidence or class info
            if not hasattr(box, 'conf') or not hasattr(box, 'cls'):
                continue
                
            # Get detection info
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Skip if confidence is too low
            if conf < conf_thresh:
                continue
                
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Get track ID if available
            track_id = -1
            if hasattr(box, 'id') and box.id is not None:
                track_id = int(box.id[0])
            else:
                # If no track ID, use box coordinates as a unique identifier
                track_id = f"box_{cls_id}_{x1}_{y1}_{x2}_{y2}"
            
            # Determine color based on whether object has been counted
            color = (0, 255, 0)  # Green for counted objects
            if track_id in track_memory:
                color = (0, 255, 255)  # Yellow for already counted objects
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Add the object to memory if not already tracked
            if track_id not in track_memory:
                track_memory[track_id] = {
                    "id": track_id,
                    "class": cls_id,
                    "counted": True,
                    "first_seen": time.time()
                }
                
            # Add label to the frame
            if hasattr(tracks[0], 'names') and tracks[0].names is not None:
                cls_name = tracks[0].names[cls_id]
            else:
                cls_name = f"Class {cls_id}"
                
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
    except Exception as e:
        print(f"Error in process_total_counting: {e}")
    
    return annotated_frame, track_memory

# Function to scale line coordinates based on video dimensions
def scale_line_coordinates(original_coords, original_dims, new_dims):
    if not original_coords or not original_dims or not new_dims:
        return original_coords
    
    orig_width, orig_height = original_dims
    new_width, new_height = new_dims
    
    scaled_coords = []
    for x, y in original_coords:
        scaled_x = int((x / orig_width) * new_width)
        scaled_y = int((y / orig_height) * new_height)
        scaled_coords.append((scaled_x, scaled_y))
    
    return scaled_coords

# Function to update sidebar based on active tab
def update_sidebar_for_tab(tab_name):
    st.sidebar.empty()
    st.sidebar.header(f"Pengaturan {tab_name}")

    if tab_name == "Video":
        # Video upload
        uploaded_file = st.sidebar.file_uploader(
            "Pilih video...", type=["mp4", "avi", "mov"], key="video_uploader"
        )

        # Line counter toggle
        use_line = st.sidebar.checkbox(
            "Gunakan Garis Penghitung", value=st.session_state.use_line, key="use_line_toggle"
        )

        # Update session state for line toggle
        if use_line != st.session_state.use_line:
            st.session_state.use_line = use_line

        # Reset line coordinates when video dimensions change
        if uploaded_file and st.session_state.video_dimensions:
            width, height = st.session_state.video_dimensions
            
            # Check if we need to reset line coordinates
            if "previous_video_dims" not in st.session_state:
                st.session_state.previous_video_dims = (width, height)
                st.session_state.line_coordinates = [
                    (int(width * 0.1), int(height * 0.5)),
                    (int(width * 0.9), int(height * 0.5))
                ]
            elif st.session_state.previous_video_dims != (width, height):
                # Video dimensions have changed, scale the line
                old_width, old_height = st.session_state.previous_video_dims
                
                # Get current line positions as percentages
                x1_percent = st.session_state.line_coordinates[0][0] / old_width
                y1_percent = st.session_state.line_coordinates[0][1] / old_height
                x2_percent = st.session_state.line_coordinates[1][0] / old_width
                y2_percent = st.session_state.line_coordinates[1][1] / old_height
                
                # Apply percentages to new dimensions
                st.session_state.line_coordinates = [
                    (int(x1_percent * width), int(y1_percent * height)),
                    (int(x2_percent * width), int(y2_percent * height))
                ]
                
                # Ensure coordinates are within bounds
                st.session_state.line_coordinates = [
                    (min(max(0, x), width), min(max(0, y), height))
                    for x, y in st.session_state.line_coordinates
                ]
                
                # Update previous dimensions
                st.session_state.previous_video_dims = (width, height)
        
        # Set default line coordinates for first run
        if "line_coordinates" not in st.session_state:
            default_width = 1280  # Default width if no video
            default_height = 720  # Default height if no video
            st.session_state.line_coordinates = [
                (int(default_width * 0.1), int(default_height * 0.5)),
                (int(default_width * 0.9), int(default_height * 0.5))
            ]

        return uploaded_file, 0.2, 0.3, None, use_line  # Fixed confidence at 0.2, IoU at 0.3

    else:  # Image tab
        # Image upload
        uploaded_image = st.sidebar.file_uploader(
            "Pilih gambar...",
            type=["jpg", "jpeg", "png", "bmp"],
            key="image_uploader",
        )

        # Visualization options
        show_labels = st.sidebar.checkbox("Tampilkan Label", value=True)
        show_conf = st.sidebar.checkbox("Tampilkan Skor Kepercayaan", value=True)

        return uploaded_image, 0.25, 0.45, (show_labels, show_conf), None  # Fixed confidence at 0.25, IoU at 0.45

# Function to update line coordinates
def update_line_coordinates():
    st.session_state.line_coordinates = [
        (st.session_state.x1, st.session_state.y1),
        (st.session_state.x2, st.session_state.y2)
    ]

# Helper function to safely get class name from model names dictionary
def get_class_name(model_names, cls_id):
    """
    Safely get class name from model names dictionary
    
    Args:
        model_names: Dictionary mapping class IDs to class names
        cls_id: Class ID to look up
        
    Returns:
        Class name as string
    """
    if not isinstance(model_names, dict):
        return f"Class {cls_id}"
        
    # Try to get class name directly
    cls_name = model_names.get(cls_id, None)
    if cls_name is not None:
        return cls_name
        
    # If cls_id is a string that might be a number, try converting it
    if isinstance(cls_id, str) and cls_id.isdigit():
        cls_name = model_names.get(int(cls_id), None)
        if cls_name is not None:
            return cls_name
            
    # If cls_id is a number, try as string
    if isinstance(cls_id, (int, float)):
        cls_name = model_names.get(str(cls_id), None)
        if cls_name is not None:
            return cls_name
            
    # Fall back to original class ID
    return f"Class {cls_id}"

# Function to clean up temporary files
def cleanup_temp_files(file_paths):
    """
    Clean up temporary files
    
    Args:
        file_paths: List of file paths to remove
    """
    for file_path in file_paths:
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                print(f"Removed temporary file: {file_path}")
        except Exception as e:
            print(f"Error removing temporary file {file_path}: {e}")

# Register cleanup handler for when the app is closed or restarted
def on_app_close():
    # Clean up any temporary files
    if "temp_files" in st.session_state and st.session_state.temp_files:
        cleanup_temp_files(st.session_state.temp_files)
        st.session_state.temp_files = []

# Register the cleanup function to run when the app is closed
atexit.register(on_app_close)

# Video Processing Tab
with tab1:
    st.session_state.current_tab = "Video"

    # Get video sidebar inputs
    uploaded_file, conf_thresh, _, _, use_line = update_sidebar_for_tab("Video")

    st.header("Penghitungan Objek Video")

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
            st.session_state.video_dimensions = (frame_width, frame_height)

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
                    st.subheader("Set Garis Penghitung")

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
                        caption="Garis Penghitung",
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
                    st.subheader("Pengaturan Proses")

                    # Display current line coordinates
                    st.write("Koordinat Garis Saat Ini:")
                    st.write(f"Start: {st.session_state.line_start}")
                    st.write(f"End: {st.session_state.line_end}")

                    # Set line points for counting
                    line_points = [
                        st.session_state.line_start,
                        st.session_state.line_end,
                    ]

                    # Process video button
                    process_button = st.button(
                        "Proses Video", type="primary", use_container_width=True
                    )
            else:
                # Simplified UI when line is disabled
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.subheader("Pratinjau Frame")

                    # Show first frame without line
                    st.image(pil_image, caption="Frame Pertama", use_container_width=True)

                with col2:
                    st.subheader("Pengaturan Proses")
                    st.info(
                        "Garis Penghitung dinonaktifkan. Semua objek akan dihitung tanpa perbedaan arah."
                    )

                    # Process video button
                    process_button = st.button(
                        "Proses Video", type="primary", use_container_width=True
                    )

                # Empty line points when line is disabled
                line_points = []

            # In the video processing section where you're processing the button click
            if process_button:
                # Load YOLO model
                with st.spinner("Memuat Model YOLO..."):
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

                # Initialize counter results in session state
                st.session_state.counter_results = {
                    "in_counts": 0,
                    "out_counts": 0,
                    "in_classes": {},
                    "out_classes": {},
                    "model_names": model.names,
                    "use_line": True
                }

                # Create a temporary directory for output
                temp_output_dir = tempfile.mkdtemp()
                out_path = os.path.join(temp_output_dir, "processed_output.mp4")

                # Progress indicators
                progress_container = st.container()

                with progress_container:
                    progress_text = st.empty()
                    progress_bar = st.progress(0)
                    progress_text.text("Mengolah frame video...")

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
                
                # Create a dictionary to track unique objects by ID to prevent duplicate counting
                unique_object_tracker = {}
                
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
                        iou=0.3,
                        verbose=False,
                        tracker="custom_tracker.yaml",
                        classes=allowed_classes,  # Filter classes during detection
                    )

                    # Process frame based on counting method
                    if use_line:
                        # Process with line counter
                        im0 = counter.start_counting(im0, tracks)
                        
                        # Update session state with current counter results after each frame
                        # Create a mapping from class names to class IDs for reverse lookup
                        class_name_to_id = {}
                        for cls_id, name in model.names.items():
                            class_name_to_id[name] = cls_id
                            
                        # Convert string class names to numeric IDs for consistent storage
                        in_classes_by_id = {}
                        out_classes_by_id = {}
                        
                        for cls_name, count in dict(counter.in_classes).items():
                            cls_id = class_name_to_id.get(cls_name, cls_name)
                            in_classes_by_id[cls_id] = count
                            
                        for cls_name, count in dict(counter.out_classes).items():
                            cls_id = class_name_to_id.get(cls_name, cls_name)
                            out_classes_by_id[cls_id] = count
                            
                        st.session_state.counter_results = {
                            "in_counts": counter.in_counts,
                            "out_counts": counter.out_counts,
                            "in_classes": in_classes_by_id,
                            "out_classes": out_classes_by_id,
                            "model_names": model.names,
                            "use_line": True,
                        }
                        
                        # Debug print counter results
                        if frame_idx % 30 == 0:  # Only print every 30 frames to avoid console spam
                            print(f"Frame {frame_idx} - Counter results: in={counter.in_counts}, out={counter.out_counts}")
                            print(f"In classes: {dict(counter.in_classes)}")
                            print(f"Out classes: {dict(counter.out_classes)}")
                    else:
                        # Process with total counter
                        im0, track_memory = process_total_counting(
                            im0,
                            tracks,
                            track_memory if "track_memory" in locals() else {},
                            conf_thresh,
                        )

                        # Only count unique objects based on their track ID
                        if tracks and len(tracks) > 0 and hasattr(tracks[0], "boxes"):
                            boxes = tracks[0].boxes
                            if boxes is not None and len(boxes) > 0 and hasattr(boxes, "id"):
                                for i, box in enumerate(boxes):
                                    if not hasattr(box, "id") or box.id is None:
                                        continue
                                        
                                    track_id = int(box.id[0])
                                    cls = int(box.cls[0])
                                    class_name = get_class_name(model.names, cls)
                                    
                                    # Only count this object if we haven't seen this ID before
                                    if track_id not in unique_object_tracker:
                                        unique_object_tracker[track_id] = {
                                            "class": cls,
                                            "class_name": class_name,
                                            "first_seen": frame_idx
                                        }
                                        # Update the final counts that will be saved to DB
                                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                                        total_detections += 1

                    # Write processed frame to output video
                    out.write(im0)

                    # Update progress
                    frame_idx += 1
                    progress_bar.progress(frame_idx / frame_count)

                # Release resources
                cap.release()
                out.release()

                # Add video path to temp files for later cleanup
                if "temp_files" not in st.session_state:
                    st.session_state.temp_files = []
                st.session_state.temp_files.append(video_path)

                # Prepare final counts for database
                if use_line and hasattr(st.session_state, 'counter_results') and st.session_state.counter_results is not None:
                    # For line-based counting, use the counter results
                    in_classes = st.session_state.counter_results.get("in_classes", {}) or {}
                    out_classes = st.session_state.counter_results.get("out_classes", {}) or {}
                    
                    # Calculate total detections (sum of in and out counts)
                    total_detections = st.session_state.counter_results.get("in_counts", 0) + st.session_state.counter_results.get("out_counts", 0)
                    
                    # Debug print
                    print(f"Final counter results: in={st.session_state.counter_results.get('in_counts', 0)}, out={st.session_state.counter_results.get('out_counts', 0)}")
                    print(f"In classes: {in_classes}")
                    print(f"Out classes: {out_classes}")
                    print(f"Total detections: {total_detections}")
                    
                    # Convert class IDs to class names in the in/out classes
                    named_in_classes = {}
                    named_out_classes = {}
                    model_names = st.session_state.counter_results.get("model_names", {})
                    
                    for cls, count in in_classes.items():
                        # Get class name and ensure consistent format
                        if isinstance(cls, str) and cls.startswith("Class "):
                            cls_name = cls[6:]  # Remove "Class " prefix
                        else:
                            cls_name = get_class_name(model_names, cls)
                            if cls_name.startswith("Class "):
                                cls_name = cls_name[6:]  # Remove "Class " prefix
                        
                        named_in_classes[cls_name] = count
                        
                    for cls, count in out_classes.items():
                        # Get class name and ensure consistent format
                        if isinstance(cls, str) and cls.startswith("Class "):
                            cls_name = cls[6:]  # Remove "Class " prefix
                        else:
                            cls_name = get_class_name(model_names, cls)
                            if cls_name.startswith("Class "):
                                cls_name = cls_name[6:]  # Remove "Class " prefix
                        
                        named_out_classes[cls_name] = count
                    
                    # Structure the class counts as requested by the user
                    structured_class_counts = {
                        "in": named_in_classes,
                        "out": named_out_classes
                    }
                    
                    # Save to database only if we have detections
                    if total_detections > 0:
                        save_detection_session(
                            file_name=uploaded_file.name,
                            file_type="video",
                            total_detections=total_detections,
                            class_counts=structured_class_counts,
                            video_dimensions=st.session_state.video_dimensions
                        )
                    else:
                        print("Warning: No detections to save to database")
                else:
                    # For non-line based counting, use the unique object tracker
                    total_detections = sum(class_counts.values())
                    
                    # Convert class IDs to class names
                    named_class_counts = {}
                    for cls, count in class_counts.items():
                        # Remove the "Class " prefix if it exists
                        if isinstance(cls, str) and cls.startswith("Class "):
                            cls_name = cls
                        else:
                            cls_name = get_class_name(model.names, cls)
                            # Ensure consistent format by removing "Class " prefix
                            if cls_name.startswith("Class "):
                                cls_name = cls_name[6:]
                        
                        named_class_counts[cls_name] = count
                    
                    # Save to database
                    save_detection_session(
                        file_name=uploaded_file.name,
                        file_type="video",
                        total_detections=total_detections,
                        class_counts=named_class_counts,
                        video_dimensions=st.session_state.video_dimensions
                    )

                # Clear progress indicators
                progress_text.empty()
                progress_bar.empty()
                progress_container.success("Proses selesai!")

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
                st.header("Hasil Proses")

                # Check if line was used for this result
                if st.session_state.counter_results.get("use_line", True):
                    # Display directional counting metrics
                    st.subheader("Hasil Penghitungan Arah")
                    col_in, col_out = st.columns(2)
                    col_in.metric(
                        "Objek Masuk", st.session_state.counter_results["in_counts"]
                    )
                    col_out.metric(
                        "Objek Keluar", st.session_state.counter_results["out_counts"]
                    )

                    # Display class details in an expandable section
                    with st.expander("Detail Kelas", expanded=True):
                        col_in_detail, col_out_detail = st.columns(2)

                        with col_in_detail:
                            st.write("**Kelas Masuk:**")
                            for cls, count in st.session_state.counter_results[
                                "in_classes"
                            ].items():
                                cls_name = get_class_name(st.session_state.counter_results["model_names"], cls)
                                st.write(f"- {cls_name}: {count}")

                        with col_out_detail:
                            st.write("**Kelas Keluar:**")
                            for cls, count in st.session_state.counter_results[
                                "out_classes"
                            ].items():
                                cls_name = get_class_name(st.session_state.counter_results["model_names"], cls)
                                st.write(f"- {cls_name}: {count}")
                else:
                    # Display total counting metrics
                    st.subheader("Jumlah Objek Total")
                    st.metric(
                        "Jumlah Objek Terdeteksi",
                        st.session_state.counter_results["total_counts"],
                    )

                    # Display class details
                    with st.expander("Detail Kelas", expanded=True):
                        class_counts = st.session_state.counter_results["class_counts"]
                        model_names = st.session_state.counter_results["model_names"]

                        for cls, count in class_counts.items():
                            cls_name = get_class_name(model_names, cls)
                            st.write(f"- {cls_name}: {count}")

                # Display the processed video
                st.subheader("Video Hasil Proses")
                st.video(st.session_state.processed_video)

                # Offer download button
                with open(st.session_state.processed_video, "rb") as file:
                    st.download_button(
                        label="Unduh Video Hasil Proses",
                        data=file,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )

    else:
        # Display instruction when no file is uploaded
        st.info("Silakan unggah file video untuk memulai.")
        st.markdown(
            """
        ### Cara menggunakan aplikasi ini:
        1. Unggah file video menggunakan sidebar
        2. Pilih apakah ingin menggunakan garis penghitung:
           - Dengan garis: Hitung objek yang melewati garis dalam kedua arah
           - Tanpa garis: Hitung semua objek dalam video
        3. Jika menggunakan garis, atur posisi garis penghitung
        4. Klik 'Proses Video' untuk memulai penghitungan objek
        5. Lihat hasil dan unduh video hasil proses
        """
        )

# Image Processing Tab
with tab2:
    st.session_state.current_tab = "Image"

    # Get image sidebar inputs
    uploaded_image, img_conf_thresh, _, vis_options, _ = update_sidebar_for_tab("Image")

    if vis_options:
        show_labels, show_conf = vis_options

    st.header("Deteksi Objek Gambar")

    if uploaded_image:
        # Save uploaded image temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tfile.write(uploaded_image.read())
        image_path = tfile.name
        tfile.close()

        # Display original image
        img = Image.open(image_path)
        st.subheader("Gambar Asli")
        st.image(img, caption="Gambar Unggahan", use_container_width=True)

        # Process image button
        process_image_button = st.button(
            "Deteksi Objek", type="primary", use_container_width=True, key="detect_btn"
        )

        if process_image_button:
            # Load YOLO model
            with st.spinner("Memuat Model YOLO dan mengolah gambar..."):
                model = YOLO("yolo11s.pt")  # Change this to your trained model path

                # Run detection
                results = model(
                    image_path,
                    conf=img_conf_thresh,
                    iou=0.45,
                    verbose=False,
                    classes=[1, 2, 3, 5, 7],
                )

                # Convert cv2 image format for visualization
                img_cv = cv2.imread(image_path)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

                # Initialize detection counts
                total_detections = 0
                class_counts = {}
                
                # Track unique objects by ID to prevent duplicate counting
                unique_objects = set()
                
                # Process and count objects by class
                if len(results) > 0 and hasattr(results[0], 'boxes') and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    # Count objects by class (only counting unique objects)
                    for box in boxes:
                        cls = int(box.cls[0])
                        class_name = get_class_name(model.names, cls)
                        
                        # For images, we don't have tracking IDs, so use box coordinates as a unique identifier
                        # This is a simplified approach since we're processing a single image
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_id = f"{cls}_{x1}_{y1}_{x2}_{y2}"
                        
                        if box_id not in unique_objects:
                            unique_objects.add(box_id)
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1
                            total_detections += 1
                
                # Save detection results to database
                if total_detections > 0:
                    # Convert class IDs to class names for database
                    named_class_counts = {}
                    for cls, count in class_counts.items():
                        # Ensure consistent format by removing "Class " prefix
                        if isinstance(cls, str) and cls.startswith("Class "):
                            cls_name = cls[6:]
                        else:
                            cls_name = get_class_name(model.names, cls)
                            if cls_name.startswith("Class "):
                                cls_name = cls_name[6:]
                        
                        named_class_counts[cls_name] = count
                    
                    # Get image dimensions
                    img_height, img_width = img_cv.shape[:2]
                    
                    # Save to database
                    save_detection_session(
                        file_name=uploaded_image.name,
                        file_type="image",
                        total_detections=total_detections,
                        class_counts=named_class_counts,
                        video_dimensions=(img_width, img_height)
                    )

                # Create a temporary directory for output
                temp_output_dir = tempfile.mkdtemp()
                result_path = os.path.join(temp_output_dir, "detected_image.jpg")

                # Track temporary files for cleanup
                if "temp_files" not in st.session_state:
                    st.session_state.temp_files = []
                st.session_state.temp_files.append(image_path)
                st.session_state.temp_files.append(temp_output_dir)

                # Draw bounding boxes on the image
                for box in results[0].boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Get class and confidence
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Get class name
                    cls_name = get_class_name(model.names, cls_id)

                    # Draw bounding box
                    color = (255, 0, 0)  # Red for all boxes
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

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
                            img_cv,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color,
                            -1,
                        )

                        # Add text
                        cv2.putText(
                            img_cv,
                            label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            2,
                        )

                # Save the processed image
                cv2.imwrite(
                    result_path, cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                )

                # Display results
                st.subheader("Hasil Deteksi")

                # Show object counts
                st.write("**Objek Terdeteksi:**")

                # Create a nice grid of metrics
                if class_counts:
                    cols = st.columns(min(3, len(class_counts)))
                    for i, (cls_name, count) in enumerate(class_counts.items()):
                        cols[i % len(cols)].metric(f"{cls_name}", f"{count}")
                else:
                    st.warning(
                        "Tidak ada objek terdeteksi. Coba atur ambang kepercayaan."
                    )

                # Show the processed image
                st.image(
                    img_cv,
                    caption="Hasil Deteksi",
                    use_container_width=True,
                )

                # Offer download button
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Unduh Gambar Hasil Deteksi",
                        data=file,
                        file_name="detected_image.jpg",
                        mime="image/jpeg",
                        key="download_img",
                    )
    else:
        # Display instruction when no file is uploaded
        st.info("Silakan unggah file gambar untuk memulai deteksi.")
        st.markdown(
            """
        ### Cara menggunakan deteksi gambar:
        1. Unggah file gambar menggunakan sidebar
        2. Atur ambang kepercayaan dan IoU untuk mengontrol sensitivitas deteksi
        3. Konfigurasikan opsi visualisasi (label, skor kepercayaan)
        4. Klik 'Deteksi Objek' untuk memulai deteksi
        5. Lihat hasil dan unduh gambar hasil deteksi
        """
        )

# History tab
with tab3:
    st.session_state.current_tab = "History"
    st.header("Riwayat Penghitungan")

    # Add filters
    st.subheader("Filter")
    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        file_type_filter = st.radio("Jenis File", ["Semua", "Video", "Gambar"])

    with filter_col2:
        limit = st.slider("Jumlah Catatan", 10, 5000, 100, 10)

    with filter_col3:
        st.write("&nbsp;")  # Add some spacing
        st.write("&nbsp;")  # Add some spacing
        show_all = st.checkbox("Tampilkan Semua Data", value=False)

    # Add a button to refresh history
    if st.button("Refresh Riwayat Data", type="primary", key="refresh_history"):
        st.rerun()

    # Add a button to clear history
    if st.button("Hapus Semua Riwayat", type="secondary", key="clear_history"):
        conn = sqlite3.connect("detection_history.db")
        c = conn.cursor()
        c.execute("DELETE FROM detection_sessions")
        conn.commit()
        conn.close()
        st.success("Semua riwayat telah dihapus!")
        st.rerun()

    # Build query based on filters
    base_query = "SELECT * FROM detection_sessions"
    where_clause = []
    
    if file_type_filter != "Semua":
        # Map UI selection to database value
        if file_type_filter == "Video":
            where_clause.append("file_type = 'video'")
        elif file_type_filter == "Gambar":
            where_clause.append("file_type = 'gambar'")
    
    query = base_query
    if where_clause:
        query += " WHERE " + " AND ".join(where_clause)
    
    query += " ORDER BY timestamp DESC"
    if not show_all:
        query += f" LIMIT {limit}"

    # Fetch and display data
    conn = sqlite3.connect("detection_history.db")
    sessions_df = pd.read_sql_query(query, conn)
    conn.close()

    if not sessions_df.empty:
        # Format class counts for better display
        def format_class_counts(counts_str):
            try:
                if not counts_str:
                    return "No data"
                    
                counts = json.loads(counts_str)
                if isinstance(counts, dict):
                    # Check if it's a structured format with 'in' and 'out'
                    if 'in' in counts and 'out' in counts:
                        in_counts = counts['in']
                        out_counts = counts['out']
                        
                        result = ""
                        
                        # Format the in counts
                        for cls, count in in_counts.items():
                            if count > 0:  # Only show classes with counts
                                result += f"- In {cls}: {count}\n"
                        
                        # Format the out counts
                        for cls, count in out_counts.items():
                            if count > 0:  # Only show classes with counts
                                result += f"- Out {cls}: {count}\n"
                        
                        return result.strip() if result else "No detections"
                    else:
                        # Regular format (non-line based)
                        return "\n".join([f"- {k}: {v}" for k, v in counts.items() if v > 0])
                return str(counts_str)
            except Exception as e:
                print(f"Error formatting class counts: {e}")
                return str(counts_str)
        
        # Apply the formatting to the class_counts column
        sessions_df['class_counts_formatted'] = sessions_df['class_counts'].apply(format_class_counts)
        
        # Create a display dataframe with better column names
        display_df = sessions_df.copy()
        display_df.rename(columns={
            'id': 'ID',
            'timestamp': 'Waktu',
            'file_name': 'Nama File',
            'file_type': 'Jenis File',
            'total_detections': 'Total Objek',
            'class_counts_formatted': 'Detail Kelas',
            'video_dimensions': 'Dimensi',
            'settings_info': 'Info Tambahan'
        }, inplace=True)
        
        # Select and reorder columns for display
        columns_to_display = ['ID', 'Waktu', 'Nama File', 'Jenis File', 'Total Objek', 'Detail Kelas', 'Dimensi']
        display_df = display_df[columns_to_display]
        
        st.dataframe(display_df, use_container_width=True)

        # Allow downloading the session history as CSV
        csv = sessions_df.to_csv(index=False)
        st.download_button(
            label="Unduh Riwayat Sesi",
            data=csv,
            file_name="detection_sessions.csv",
            mime="text/csv",
        )
    else:
        st.info("Belum ada sesi pengolahan. Proses beberapa video atau gambar terlebih dahulu.")
