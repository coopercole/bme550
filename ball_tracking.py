# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import matplotlib.pyplot as plt


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", default="high_n.MOV", 
	help="path to the (optional) video file")
args = vars(ap.parse_args())


# define the lower and upper boundaries of the "yellow"
# ball in the HSV color space, then initialize the
yellowLower = (20, 200, 190)
yellowUpper = (28, 255, 248)
# list of tracked points
pts = []
# list of speeds
speeds = []


# video capture object
vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)
# Get the frame rate
frame_rate = vs.get(cv2.CAP_PROP_FPS)


# keep looping
while True:
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[1] if args.get("video", False) else frame
	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break
	# resize the frame, blur it, and convert it to the HSV
	# color space
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
	# construct a mask for the color "yellow", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, yellowLower, yellowUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)
	# cv2.imshow("mask", mask)
	
	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None
	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((t, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(t), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
	# update the points queue
	pts.append(center)
	
    	# loop over the set of tracked points
	frames_per_iteration = 1
	for i in range(frames_per_iteration, len(pts), frames_per_iteration):
		# if either of the tracked points are None, ignore them
		if pts[i - frames_per_iteration] is None or pts[i] is None:
			continue
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - frames_per_iteration], pts[i], (0, 0, 255), thickness)
		distance = np.array(pts[i]) - np.array(pts[i-frames_per_iteration])
		speed = distance * frame_rate / frames_per_iteration
		# print(f'Speed: {speed} pixels/second')
		speed_in_meters_per_second = speed / 811.46
		# print(f'Speed: {speed_in_meters_per_second} m/s')
		speeds.append(speed_in_meters_per_second)
	# show the frame to our screen
	# cv2.imshow("Frame", frame)
	# key = cv2.waitKey(1) & 0xFF
	# # if the 'q' key is pressed, stop the loop
	# if key == ord("q"):
	# 	break


vs.release()
# close all windows
cv2.destroyAllWindows()

speeds = np.vstack(speeds)

t = np.linspace(0, len(speeds), len(speeds))
print(speeds.shape)
vx = speeds[:,0]
vy = speeds[:,1]

# Fit a curve to the data
degree = 7
coefficients = np.polyfit(t, vx, degree)
polynomial = np.poly1d(coefficients)
vxs = polynomial(t)
coefficients = np.polyfit(t, vy, degree)
polynomial = np.poly1d(coefficients)
vys = polynomial(t)

plt.plot(t, vxs)
plt.plot(t, vys)
plt.xlabel('Frame')
plt.ylabel('Speed (m/s)')
plt.show()

# After the main loop
x_coords = [pt[0] for pt in pts if pt is not None]
y_coords = [pt[1] for pt in pts if pt is not None]

plt.scatter(x_coords, y_coords)
plt.xlabel('X')
plt.ylabel('Y')
plt.gca().invert_yaxis()  # Invert y axis to match image coordinates
plt.show()


# window_size = 750
# window = np.ones(window_size)/window_size
# smoothed_speeds = np.convolve(speeds, window, mode='valid')

# plt.plot(smoothed_speeds)
# plt.xlabel('Frame')
# plt.ylabel('Speed (m/s)')
# plt.show()
