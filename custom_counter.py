# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import defaultdict

import cv2

from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")

from shapely.geometry import LineString, Point, Polygon


class CustomCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""

        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5
        self.use_line = True  # Default to use line counter

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = None  # Classes names
        self.annotator = None  # Annotator

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.counting_list = []
        self.count_txt_thickness = 0
        self.count_txt_color = (0, 0, 0)
        self.count_color = (255, 255, 255)

        # Classes counting information
        self.in_classes = defaultdict(int)
        self.out_classes = defaultdict(int)

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = (0, 255, 0)
        
        # Debug addition - explicitly store allowed classes
        self.allowed_classes = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        count_txt_thickness=1,
        count_txt_color=(0, 0, 0),
        count_color=(255, 255, 255),
        track_color=(0, 255, 0),
        region_thickness=5,
        line_dist_thresh=15,
        use_line=None,  # Added the use_line parameter
        allowed_classes=None,  # Added allowed_classes parameter
    ):


        """
        Configures the Counter's image, bounding box line thickness, and counting region points.

        Args:
            line_thickness (int): Line thickness for bounding boxes.
            view_img (bool): Flag to control whether to display the video stream.
            view_in_counts (bool): Flag to control whether to display the incounts on video stream.
            view_out_counts (bool): Flag to control whether to display the outcounts on video stream.
            reg_pts (list): Initial list of points defining the counting region.
            classes_names (dict): Classes names
            track_thickness (int): Track thickness
            draw_tracks (Bool): draw tracks
            count_txt_thickness (int): Text thickness for object counting display
            count_txt_color (RGB color): count text color value
            count_color (RGB color): count text background color value
            count_reg_color (RGB color): Color of object counting region
            track_color (RGB color): color for tracks
            region_thickness (int): Object counting Region thickness
            line_dist_thresh (int): Euclidean Distance threshold for line counter
            use_line (bool, optional): Force use of line counter instead of region when True
            allowed_classes (set, optional): Set of class IDs to track and count
        """
        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        self.names = classes_names
        
        # Update allowed classes if provided
        if allowed_classes is not None:
            self.allowed_classes = set(allowed_classes)

        # Update use_line if provided
        if use_line is not None:
            self.use_line = use_line

        # Debug print
        print(f"Allowed classes: {self.allowed_classes}")
        allowed_class_names = [self.names.get(cls, f"Unknown-{cls}") for cls in self.allowed_classes]
        print(f"Allowed class names: {allowed_class_names}")

        # Region and line selection
        if self.use_line or len(reg_pts) == 2:
            print("Line Counter Initiated.")
            self.reg_pts = reg_pts[:2] if len(reg_pts) > 2 else reg_pts  # Ensure we only take 2 points if use_line is True
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) == 4:
            print("Region Counter Initiated.")
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print("Invalid Region points provided, region_points can be 2 or 4")
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_thickness = count_txt_thickness
        self.count_txt_color = count_txt_color
        self.count_color = count_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        
        # Debug print
        print(f"Allowed classes: {self.allowed_classes}")
        allowed_class_names = [self.names.get(cls, f"Unknown-{cls}") for cls in self.allowed_classes]
        print(f"Allowed class names: {allowed_class_names}")

    def mouse_event_for_region(self, event, x, y, flags, params):
        """
        This function is designed to move region with mouse events in a real-time video stream.

        Args:
            event (int): The type of mouse event (e.g., cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN, etc.).
            x (int): The x-coordinate of the mouse pointer.
            y (int): The y-coordinate of the mouse pointer.
            flags (int): Any flags associated with the event (e.g., cv2.EVENT_FLAG_CTRLKEY,
                cv2.EVENT_FLAG_SHIFTKEY, etc.).
            params (dict): Additional parameters you may want to pass to the function.
        """


        # global is_drawing, selected_point
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if isinstance(point, (tuple, list)) and len(point) >= 2:
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                        self.selected_point = i
                        self.is_drawing = True
                        break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                if self.use_line or len(self.reg_pts) == 2:
                    self.counting_region = LineString(self.reg_pts)
                else:
                    self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""
                    
        try:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            
            # Safety check for track IDs - handle None case
            if hasattr(tracks[0].boxes, 'id') and tracks[0].boxes.id is not None:
                track_ids = tracks[0].boxes.id.int().cpu().tolist()
            else:
                # If no track IDs, create sequential IDs
                print("Warning: No tracking IDs found, creating sequential IDs")
                track_ids = list(range(len(boxes)))
        except (AttributeError, IndexError) as e:
            print(f"Error processing tracks: {e}")
            return  # Exit if we can't process the tracks


        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=self.region_color, thickness=self.region_thickness)

        # Extract tracks
        for box, track_id, cls in zip(boxes, track_ids, clss):
            # Skip classes that are not in the allowed list
            if int(cls) not in self.allowed_classes:
                # Debug print for skipped classes
                print(f"Skipping class {int(cls)} ({self.names.get(int(cls), 'Unknown')})")
                continue

            class_name = self.names[int(cls)]
            print(f"Processing {class_name} (ID: {track_id})")

            self.annotator.box_label(
                box, label=str(track_id) + ":" + class_name, color=colors(int(cls), True)
            )  # Draw bounding box

            # Draw Tracks
            track_line = self.track_history[track_id]
            track_line.append((float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)))
            if len(track_line) > 30:
                track_line.pop(0)

            # Draw track trails
            if self.draw_tracks:
                self.annotator.draw_centroid_and_tracks(
                    track_line, color=self.track_color, track_thickness=self.track_thickness
                )

            # Count objects based on direction
            # Use more points for reliable direction detection
            min_track_points = 5

            # For region-based counting
            if not self.use_line and len(self.reg_pts) == 4:
                # Check if object is in the counting region
                if self.counting_region.contains(Point(track_line[-1])):
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        
                        # Determine direction if enough track history exists
                        if len(track_line) >= min_track_points:
                            # Calculate overall movement direction
                            start_x = track_line[-min_track_points][0]
                            current_x = track_line[-1][0]
                            
                            if current_x > start_x:  # Moving right
                                self.in_counts += 1
                                self.in_classes[class_name] += 1
                                print(f"IN: {class_name} (moving right)")
                            else:  # Moving left
                                self.out_counts += 1
                                self.out_classes[class_name] += 1
                                print(f"OUT: {class_name} (moving left)")
                        else:
                            # Fallback if not enough track history
                            # Use position relative to region center
                            if box[0] < self.counting_region.centroid.x:
                                self.out_counts += 1
                                self.out_classes[class_name] += 1
                                print(f"OUT: {class_name} (position-based)")
                            else:
                                self.in_counts += 1
                                self.in_classes[class_name] += 1
                                print(f"IN: {class_name} (position-based)")

            # For line-based counting
            elif self.use_line or len(self.reg_pts) == 2:
                distance = Point(track_line[-1]).distance(self.counting_region)
                if distance < self.line_dist_thresh:
                    if track_id not in self.counting_list:
                        self.counting_list.append(track_id)
                        
                        # Determine direction if enough track history exists
                        if len(track_line) >= min_track_points:
                            # Get line orientation
                            line_dx = self.reg_pts[1][0] - self.reg_pts[0][0]
                            line_dy = self.reg_pts[1][1] - self.reg_pts[0][1]
                            
                            # Determine if line is more horizontal or vertical
                            is_horizontal = abs(line_dx) > abs(line_dy)
                            
                            if is_horizontal:
                                # For horizontal line, check vertical movement
                                start_y = track_line[-min_track_points][1]
                                current_y = track_line[-1][1]
                                if current_y < start_y:  # Moving upward
                                    self.in_counts += 1
                                    self.in_classes[class_name] += 1
                                    print(f"IN: {class_name} (moving up)")
                                else:  # Moving downward
                                    self.out_counts += 1
                                    self.out_classes[class_name] += 1
                                    print(f"OUT: {class_name} (moving down)")
                            else:
                                # For vertical line, check horizontal movement
                                start_x = track_line[-min_track_points][0]
                                current_x = track_line[-1][0]
                                if current_x > start_x:  # Moving right
                                    self.in_counts += 1
                                    self.in_classes[class_name] += 1
                                    print(f"IN: {class_name} (moving right)")
                                else:  # Moving left
                                    self.out_counts += 1
                                    self.out_classes[class_name] += 1
                                    print(f"OUT: {class_name} (moving left)")
                        else:
                            # Fallback if not enough track history
                            if box[0] < self.counting_region.centroid.x:
                                self.out_counts += 1
                                self.out_classes[class_name] += 1
                                print(f"OUT: {class_name} (position-based)")
                            else:
                                self.in_counts += 1
                                self.in_classes[class_name] += 1
                                print(f"IN: {class_name} (position-based)")

        self.display_class_counts()

        incount_label = "In:" + f"{self.in_counts}"
        outcount_label = "Out:" + f"{self.out_counts}"

        # Display counts based on user choice
        counts_label = None
        if not self.view_in_counts and not self.view_out_counts:
            counts_label = None
        elif not self.view_in_counts:
            counts_label = outcount_label
        elif not self.view_out_counts:
            counts_label = incount_label
        else:
            counts_label = incount_label + " " + outcount_label

        if counts_label is not None:
            cv2.putText(self.im0, counts_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.count_txt_color, 2)

    def display_class_counts(self):
        font_scale = 0.5
        line_height = 18
        bg_padding = 4

        # Titles for 'In' and 'Out' classes with background, swapping their positions
        title_bg_height = 25  # Height of the background rectangle for titles
        cv2.rectangle(self.im0, (self.im0.shape[1] - 220, 5), (self.im0.shape[1] - 10, 5 + title_bg_height),
                      self.count_color, -1)  # Background for "Classes Entering:"
        cv2.putText(self.im0, "Classes Entering:", (self.im0.shape[1] - 210, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    self.count_txt_color, 1)

        cv2.rectangle(self.im0, (10, 5), (220, 5 + title_bg_height), self.count_color,
                      -1)  # Background for "Classes Leaving:"
        cv2.putText(self.im0, "Classes Leaving:", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale, self.count_txt_color,
                    1)

        # Display 'In' (Entering) counts on the right with background
        y_position_in = 40  # Start below the title
        for cls, count in self.in_classes.items():
            text = f"{cls}: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            text_width, text_height = text_size[0], text_size[1]
            text_start_x = self.im0.shape[1] - 10 - text_width
            cv2.rectangle(self.im0, (text_start_x, y_position_in - line_height + bg_padding // 2),
                          (self.im0.shape[1] - 10, y_position_in + bg_padding // 2), self.count_color, -1)
            cv2.putText(self.im0, text, (text_start_x, y_position_in), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        self.count_txt_color, 1)
            y_position_in += line_height

        # Display 'Out' (Leaving) counts on the left with background
        y_position_out = 40  # Start below the title
        for cls, count in self.out_classes.items():
            text = f"{cls}: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
            cv2.rectangle(self.im0, (10, y_position_out - text_size[1] - bg_padding // 2),
                          (10 + text_size[0], y_position_out + bg_padding // 2), self.count_color, -1)
            cv2.putText(self.im0, text, (10, y_position_out), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                        self.count_txt_color, 1)
            y_position_out += line_height

    def display_frames(self):
        """Display frame."""
        if self.env_check:
            cv2.namedWindow("Object Counter")
            if len(self.reg_pts) == 4 and not self.use_line:  # only add mouse event If user drawn region and not using line
                cv2.setMouseCallback(
                    "Object Counter", self.mouse_event_for_region, {"region_points": self.reg_pts}
                )
            elif self.use_line or len(self.reg_pts) == 2:  # Also set mouse callback for line counter
                cv2.setMouseCallback(
                    "Object Counter", self.mouse_event_for_region, {"region_points": self.reg_pts}
                )
            cv2.imshow("Object Counter", self.im0)
            # Break Window
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return

    def start_counting(self, im0, tracks):
        """
        Main function to start object counting process.
        Args:
            im0 (ndarray): Current frame.
            tracks (list): List of tracks.
        Returns:
            im0 (ndarray): Annotated frame.
        """
        if not tracks or len(tracks) == 0:
            return im0
            
        self.im0 = im0
        self.annotator = Annotator(self.im0, line_width=self.tf)
        
        # Basic validation of tracks
        if not hasattr(tracks[0], 'boxes') or tracks[0].boxes is None or len(tracks[0].boxes) == 0:
            print("Warning: No valid boxes in tracks")
            # Draw only the counting line/region without processing tracks
            self.draw_counting_region()
            return self.im0
        
        # Check if tracking is correctly initialized
        if not hasattr(tracks[0].boxes, 'id') or tracks[0].boxes.id is None:
            print("Warning: Tracking IDs not available. Running in detection-only mode.")
            # You might want to run YOLO in track mode again or handle this case
        
        # Process the tracks
        try:
            self.extract_and_process_tracks(tracks)
        except Exception as e:
            print(f"Error in track processing: {e}")
            import traceback
            traceback.print_exc()
        
        # Draw the counting region regardless of tracking success
        self.draw_counting_region()
        
        # Display frame with counting information
        self.display_frames()
        
        return self.im0

    def draw_counting_region(self):
        """Draw the counting line or region on the image"""
        if isinstance(self.counting_region, LineString):
            # Draw the counting line using cv2
            line_points = list(self.counting_region.coords)
            cv2.line(
                self.im0,
                (int(line_points[0][0]), int(line_points[0][1])),
                (int(line_points[1][0]), int(line_points[1][1])),
                self.region_color,
                thickness=self.region_thickness,
            )
        elif isinstance(self.counting_region, Polygon):
            # Draw the counting region using cv2
            import numpy as np
            region_points = list(self.counting_region.exterior.coords)
            region_points_array = np.array([[int(p[0]), int(p[1])] for p in region_points], np.int32)
            region_points_array = region_points_array.reshape((-1, 1, 2))
            cv2.polylines(
                self.im0,
                [region_points_array],
                isClosed=True,
                color=self.region_color,
                thickness=self.region_thickness,
            )

if __name__ == "__main__":
    CustomCounter()