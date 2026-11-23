# Tracking fiducial markers (AprilTag / ArUco)

This example tracks objects that carry a printed fiducial marker. The *Marker Detection (ArUco/AprilTag)*
component (`MarkerDetection`) finds the markers on the grayscale image and outputs one particle per marker,
with the marker ID as particle ID, so the tracker keeps the identity of every object even when objects
cross, leave the image or are occluded.

Pipeline (`markers.swistrack`):

    InputFileAVI (markers.avi) -> ConvertInputToGray -> MarkerDetection -> MultiObjectTracking -> OutputTracksToFiles

To use a camera instead of the video, replace the `InputFileAVI` component with `InputCameraUSB`.

`markers.avi` is a short synthetic video (80 frames, 720x480) of four AprilTag 36h11 markers (IDs 3, 7, 12
and 25) moving and rotating. It was rendered with `make_test_video.py`, which you can adapt to test other
dictionaries, sizes or speeds.

## Printing markers

Use the **AprilTag 36h11** family (the default): it has 587 IDs and a large Hamming distance, so false
detections and ID confusions are very rare.

1. Get the marker images, either
   - from the official repository <https://github.com/AprilRobotics/apriltag-imgs> (folder `tag36h11`,
     files `tag36_11_00003.png` etc.). These images are tiny (10x10 pixels, one pixel per cell), so scale
     them up **without smoothing** (nearest neighbor), e.g. `convert tag36_11_00003.png -scale 5000% tag3.png`; or
   - generate them with OpenCV (Python):

         import cv2
         d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
         cv2.imwrite("tag3.png", cv2.aruco.generateImageMarker(d, 3, 800))  # cv2.aruco.drawMarker in OpenCV < 4.7

2. Print them on matte paper at a known size, and keep a **white margin** of at least one cell around the
   black border (the detector needs the contrast between the black border and its surroundings).
3. Make the markers large enough: each marker cell should cover at least 3-4 image pixels, i.e. a 36h11
   marker (8 cells including the black border) should be at least ~30 pixels wide in the image. Small
   markers can be found by lowering *Min. marker perimeter*, at the cost of speed.
4. Glue each marker flat on the object, with the marker's right edge towards the object's front: the
   particle orientation (see below) is then the heading of the object.

## Orientation

The particle orientation is the angle of the marker's x-axis (from its top-left to its top-right corner)
in image coordinates (x to the right, y down), in radians: 0 when the marker is upright, increasing
when the marker turns clockwise on screen.

According to the OpenCV documentation, OpenCV defines the top-left corner of AprilTag markers differently
from the official AprilTag images: a marker printed from <https://github.com/AprilRobotics/apriltag-imgs>
appears rotated by 180 degrees, i.e. it has orientation +-pi when the printed image is upright. Markers
generated with OpenCV (as above) have orientation 0 when upright. Either way, the offset is constant, so just glue all markers the same way.

## Settings

- **Dictionary**: must match the printed markers. AprilTag 36h11 is recommended; ArUco 4x4 markers are
  smaller (fewer cells), and can be detected from farther away, but are more error prone.
- **Corner refinement**: *Sub-pixel* (default) is fast and accurate. *AprilTag* is the most accurate on
  blurry images, but considerably slower: use it only if the frame rate allows.
- **Filter by ID**: discards markers outside [Min. ID, Max. ID], for instance to ignore markers on the walls.
- **Show rejected candidates**: draws the square shapes that were not decoded as markers (red), which helps
  to diagnose lighting, focus or size problems.

Marker detection requires OpenCV >= 4.7 (objdetect module), or an older OpenCV with the opencv_contrib
`aruco` module (e.g. `libopencv-dev` on Ubuntu 22.04/24.04). Otherwise the component reports that SwisTrack
was built without ArUco support.
