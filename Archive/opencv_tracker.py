# opencv_tracker.py
# @author: Austin Shin

import pyrealsense2 as rs
import numpy as np
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import Realsense

cam = Realsense()
profile = cam.pipeline.start(cam.config)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
align_to = rs.stream.color
align = rs.align(align_to)

initBB = None # bounding box
fps = None # frame rate

# built-in object trackers
OPENCV_OBJECT_TRACKERS = {
  "csrt": cv2.TrackerCSRT_create,
  "kcf": cv2.TrackerKCF_create,
  "boosting": cv2.TrackerBoosting_create,
  "mil": cv2.TrackerMIL_create,
  "tld": cv2.TrackerTLD_create,
  "medianflow": cv2.TrackerMedianFlow_create,
  "mosse": cv2.TrackerMOSSE_create
}

text = raw_input("Please pick an OpenCV object tracker: csrt, kcf, boosting, mil, tld, medianflow, mosse.\n")
text = text.lower()
while (text not in OPENCV_OBJECT_TRACKERS):
  text = raw_input("That tracker does not exist. Please pick again: csrt, kcf, boosting, mil, tld, medianflow, mosse.\n")
  text = text.lower()

tracker = OPENCV_OBJECT_TRACKERS[text]()
print("Initializing tracking system...")

try:
  print("Press s to put bounding box around desired object.")
  while True:
    # get frame from camera
    frames = cam.pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not aligned_depth_frame or not color_frame:
      continue

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    frame = color_image
    (H, W) = frame.shape[:2]

    if initBB is not None: # if user placed initial bounding box
      (success, box) = tracker.update(frame)

      if success: # object is still tracked in image
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2) # bounding box

      fps.update()
      fps.stop()

      info = [
        ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps())), ]

      # write above info list into image
      for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i*20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key & 0xFF == ord('s'): # press s to put initial bounding box
      initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

      tracker.init(frame, initBB)
      fps = FPS().start()

    elif key & 0xFF == ord('q') or key == 27: # press q to quit
      cv2.destroyAllWindows()
      break

finally:
  cam.pipeline.stop()
