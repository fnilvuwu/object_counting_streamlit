import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
import custom_counter
import os
import time
from PIL import Image, ImageDraw

# Set page config
st.set_page_config(layout="wide", page_title="YOLOv8 Object Detection & Counting")

# Initialize session state
if 'line_start' not in st.session_state:
    st.session_state.line_start = None
    st.session_state.line_end = None
    st.session_state.processed_video = None
    st.session_state.processing_complete = False
    st.session_state.counter_results = None
    st.session_state.current_tab = "Video"  # Add a state to track current tab

# Streamlit UI
st.title("YOLOv8 Object Detection & Counting")

# Create tabs for Video and Image processing
tab1, tab2 = st.tabs(["Video Processing", "Image Processing"])

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
        conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.2, 0.05, key="video_conf")
        iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.3, 0.05, key="video_iou")
        
        return uploaded_file, conf_thresh, iou_thresh, None
        
    else:  # Image tab
        # Image processing sidebar
        uploaded_image = st.sidebar.file_uploader(
            "Choose an image...", type=["jpg", "jpeg", "png", "bmp"], key="image_uploader"
        )
        
        # Adjustable confidence and IoU thresholds
        img_conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05, key="img_conf")
        img_iou_thresh = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05, key="img_iou")
        
        # Option to show labels and confidence scores
        show_labels = st.sidebar.checkbox("Show Labels", value=True)
        show_conf = st.sidebar.checkbox("Show Confidence Scores", value=True)
        
        return uploaded_image, img_conf_thresh, img_iou_thresh, (show_labels, show_conf)

# Video Processing Tab
with tab1:
    st.session_state.current_tab = "Video"
    
    # Get video sidebar inputs
    uploaded_file, conf_thresh, iou_thresh, _ = update_sidebar_for_tab("Video")
    
    st.header("Video Object Counting")
    
    if uploaded_file:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
        tfile.close()
        
        # Display original video
        st.video(video_path)
        
        # Extract first frame for line drawing
        cap = cv2.VideoCapture(video_path)
        success, first_frame = cap.read()
        cap.release()
        
        if success:
            # Store video dimensions
            frame_height, frame_width = first_frame.shape[:2]
            
            # Convert to RGB for display
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(first_frame_rgb)
            
            # Initialize line coordinates if not set
            if st.session_state.line_start is None:
                st.session_state.line_start = (int(frame_width * 0.2), int(frame_height * 0.5))
                st.session_state.line_end = (int(frame_width * 0.8), int(frame_height * 0.5))
            
            # Create columns for the UI layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("Set Counting Line")
                
                # Create a copy of the image with the line drawn on it
                image_with_line = pil_image.copy()
                draw = ImageDraw.Draw(image_with_line)
                
                # Draw the line
                draw.line([st.session_state.line_start, st.session_state.line_end], 
                          fill=(255, 0, 0), width=5)
                
                # Draw the start point (green)
                draw.ellipse([(st.session_state.line_start[0]-5, st.session_state.line_start[1]-5),
                              (st.session_state.line_start[0]+5, st.session_state.line_start[1]+5)], 
                             fill=(0, 255, 0))
                
                # Draw the end point (blue)
                draw.ellipse([(st.session_state.line_end[0]-5, st.session_state.line_end[1]-5),
                              (st.session_state.line_end[0]+5, st.session_state.line_end[1]+5)], 
                             fill=(0, 0, 255))
                
                # Display the image with line
                st.image(image_with_line, caption="Counting Line", use_container_width=True)
                
                # Manual coordinate inputs with sliders for easier adjustment
                st.subheader("Adjust Line Position")
                    
                # Create two rows of sliders
                col_start_x, col_start_y = st.columns(2)
                col_end_x, col_end_y = st.columns(2)
                
                # Adjust start point
                new_start_x = col_start_x.slider("Start X", 0, frame_width, 
                                                st.session_state.line_start[0],
                                                key="start_x_slider")
                new_start_y = col_start_y.slider("Start Y", 0, frame_height, 
                                                st.session_state.line_start[1],
                                                key="start_y_slider")
                
                # Adjust end point
                new_end_x = col_end_x.slider("End X", 0, frame_width, 
                                            st.session_state.line_end[0],
                                            key="end_x_slider")
                new_end_y = col_end_y.slider("End Y", 0, frame_height, 
                                            st.session_state.line_end[1],
                                            key="end_y_slider")
                
                # Update line coordinates if changed
                if (new_start_x != st.session_state.line_start[0] or 
                    new_start_y != st.session_state.line_start[1] or
                    new_end_x != st.session_state.line_end[0] or
                    new_end_y != st.session_state.line_end[1]):
                    
                    st.session_state.line_start = (int(new_start_x), int(new_start_y))
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
                    st.session_state.line_end
                ]
                
                # Process video button
                process_button = st.button("Process Video", type="primary", use_container_width=True)
                
                if process_button:
                    # Load YOLO model
                    with st.spinner("Loading YOLO model..."):
                        model = YOLO("best.pt")  # Change this to your trained model path
                    
                    # Open video
                    cap = cv2.VideoCapture(video_path)
                    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(
                        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                    )
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    # Setup counter with the manually set line
                    counter = custom_counter.CustomCounter()
                    counter.set_args(
                        view_img=False, 
                        reg_pts=line_points, 
                        classes_names=model.names, 
                        draw_tracks=True
                    )
                    
                    # Create a temporary directory for output
                    temp_output_dir = tempfile.mkdtemp()
                    out_path = os.path.join(temp_output_dir, "processed_output.mp4")
                    
                    # Process video
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    progress_container = st.container()
                    
                    with progress_container:
                        progress_text = st.empty()
                        progress_bar = st.progress(0)
                        progress_text.text("Processing video frames...")
                    
                    frame_idx = 0
                    
                    # Use H.264 codec for better compatibility
                    fourcc = cv2.VideoWriter_fourcc(*'H264') if cv2.VideoWriter_fourcc(*'H264') != -1 else cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    
                    while cap.isOpened():
                        success, im0 = cap.read()
                        if not success:
                            break
                        tracks = model.track(
                            im0,
                            persist=True,
                            conf=conf_thresh,
                            iou=iou_thresh,
                            verbose=False,
                            tracker="bytetrack.yaml",
                        )
                        im0 = counter.start_counting(im0, tracks)
                        out.write(im0)
                        frame_idx += 1
                        progress_bar.progress(frame_idx / frame_count)
                    
                    cap.release()
                    out.release()
                    
                    # Clear progress indicators
                    progress_text.empty()
                    progress_bar.empty()
                    progress_container.success("Processing completed!")
                    
                    # Store results in session state
                    st.session_state.processed_video = out_path
                    st.session_state.processing_complete = True
                    st.session_state.counter_results = {
                        "in_counts": counter.in_counts,
                        "out_counts": counter.out_counts,
                        "in_classes": counter.in_classes,
                        "out_classes": counter.out_classes,
                        "model_names": model.names
                    }
                    
                    # Force page refresh to show results
                    st.rerun()
            
            # Display results section after processing is complete
            if st.session_state.processing_complete and st.session_state.processed_video:
                st.header("Processing Results")
                
                # Display counting metrics
                st.subheader("Counting Results")
                col_in, col_out = st.columns(2)
                col_in.metric("Objects In", st.session_state.counter_results["in_counts"])
                col_out.metric("Objects Out", st.session_state.counter_results["out_counts"])
                
                # Display class details in an expandable section
                with st.expander("Detailed Class Counts", expanded=True):
                    col_in_detail, col_out_detail = st.columns(2)
                    
                    with col_in_detail:
                        st.write("**In Classes:**")
                        for cls, count in st.session_state.counter_results["in_classes"].items():
                            # Fixed: No conversion attempt, using cls directly as the key or displaying it as is
                            if isinstance(st.session_state.counter_results["model_names"], dict):
                                # If model_names is a dictionary, try to get the class name
                                # Handle both integer and string keys
                                try:
                                    # Try with the original key
                                    cls_name = st.session_state.counter_results["model_names"].get(cls, cls)
                                    # If not found and cls is a string that might represent a number
                                    if cls_name == cls and cls.isdigit():
                                        cls_name = st.session_state.counter_results["model_names"].get(int(cls), cls)
                                except (ValueError, TypeError):
                                    cls_name = cls
                            else:
                                # If model_names is not a dictionary, just use the class key as is
                                cls_name = cls
                            st.write(f"- {cls_name}: {count}")
                    
                    with col_out_detail:
                        st.write("**Out Classes:**")
                        for cls, count in st.session_state.counter_results["out_classes"].items():
                            # Fixed: Same approach as above
                            if isinstance(st.session_state.counter_results["model_names"], dict):
                                try:
                                    cls_name = st.session_state.counter_results["model_names"].get(cls, cls)
                                    if cls_name == cls and cls.isdigit():
                                        cls_name = st.session_state.counter_results["model_names"].get(int(cls), cls)
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
                        mime="video/mp4"
                    )
    
    else:
        # Display instruction when no file is uploaded
        st.info("Please upload a video file to begin.")
        st.markdown("""
        ### How to use this app:
        1. Upload a video file using the sidebar
        2. Adjust the counting line position using the sliders
        3. Click 'Process Video' to start object counting
        4. View results and download the processed video
        """)

# Image Processing Tab
with tab2:
    st.session_state.current_tab = "Image"
    
    # Get image sidebar inputs
    uploaded_image, img_conf_thresh, img_iou_thresh, vis_options = update_sidebar_for_tab("Image")
    
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
        process_image_button = st.button("Detect Objects", type="primary", use_container_width=True, key="detect_btn")
        
        if process_image_button:
            # Load YOLO model
            with st.spinner("Loading YOLO model and processing image..."):
                model = YOLO("best.pt")  # Change this to your trained model path
                
                # Run detection
                results = model(
                    image_path,
                    conf=img_conf_thresh,
                    iou=img_iou_thresh,
                    verbose=False
                )
                
                # Convert cv2 image format for visualization
                img_cv = cv2.imread(image_path)
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                
                # Process and visualize results
                processed_image = img_cv.copy()
                
                # Get detection results
                boxes = results[0].boxes
                class_counts = {}
                
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
                    
                    # Draw bounding box
                    color = (255, 0, 0)  # Red for all boxes
                    cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label if enabled
                    if show_labels:
                        label = f"{cls_name}"
                        if show_conf:
                            label += f" {conf:.2f}"
                        
                        # Calculate text size
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        
                        # Fill background for text
                        cv2.rectangle(processed_image, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                        
                        # Add text
                        cv2.putText(processed_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save the processed image
                cv2.imwrite(result_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
                
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
                    st.warning("No objects detected. Try adjusting the confidence threshold.")
                
                # Show the processed image
                st.image(processed_image, caption="Detection Results", use_container_width=True)
                
                # Offer download button
                with open(result_path, "rb") as file:
                    st.download_button(
                        label="Download Processed Image",
                        data=file,
                        file_name="detected_image.jpg",
                        mime="image/jpeg",
                        key="download_img"
                    )
    else:
        # Display instruction when no file is uploaded
        st.info("Please upload an image file to begin detection.")
        st.markdown("""
        ### How to use image detection:
        1. Upload an image file using the sidebar
        2. Adjust the confidence and IoU thresholds to control detection sensitivity
        3. Configure visualization options (labels, confidence scores)
        4. Click 'Detect Objects' to start detection
        5. View results and download the processed image
        """)