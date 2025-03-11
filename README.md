# YOLOv8 Object Counting and Tracking
## Compatible with local video and RTSP streams

Object counting &amp; tracking, accept any YOLOv8 detection model and any input (local video or RTSP streams)

## How to use

python counting_show_video.py --h
usage: counting_show_video.py [-h] --model MODEL--path PATH
                                      --x_start X_START --x_end X_END --y_start Y_START --y_end Y_END 
                                      --conf CONF --iou IOU

YOLOv8 Object Counting

* options:
* -h, --help         show this help message and exit
* --model MODEL      Path to saved model
*  --path PATH        Path tp target video file
*  --x_start X_START  Fraction of the video width where the line starts (0.0 = left, 1.0 = right)
*  --x_end X_END      Fraction of the video width where the line ends (0.0 = left, 1.0 = right)
*  --y_start Y_START  Fraction of the video height where the line starts (0.0 = bottom, 1.0 = top)
*  --y_end Y_END      Fraction of the video height where the line ends (0.0 = bottom, 1.0 = top)
*  --conf CONF        Object confidence threshold (default: 0.2)
*  --iou IOU          IOU threshold for NMS (default: 0.3)

  ## Requirements

  ### - Python 3.10
  ### -Ultralytics == 8.0.239
