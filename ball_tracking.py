# import the necessary packages
import numpy as np
import pandas as pd
import cv2
import imutils
import time
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

video = "high_n.MOV"
PIXELS_PER_METER = 811.46
SHOW_PLOTS = True
WRITE_DATA_TO_CSV = True

def get_ball_hsv(frame):
		# Select ROI
	roi = cv2.selectROI(frame)
	# Crop image
	roi_cropped = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
	# Convert to HSV
	hsv_roi = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)

	# Calculate average HSV values
	average_color_per_row = np.average(hsv_roi, axis=0)
	average_color = np.average(average_color_per_row, axis=0)

	# Calculate min and max HSV values
	min_color = np.min(hsv_roi, axis=(0, 1))
	max_color = np.max(hsv_roi, axis=(0, 1))
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	return min_color, max_color

# define the lower and upper boundaries of the "yellow" ball in the HSV color space
yellowLower = (20, 200, 190)
yellowUpper = (28, 255, 248)
# list of tracked points
tracked_pts = []

# video capture object
cap = cv2.VideoCapture(video)
# allow the camera or video file to warm up
time.sleep(2.0)
# Get the frame rate
frame_rate = cap.get(cv2.CAP_PROP_FPS)
# print(f'Frame rate: {frame_rate} fps')

# Track the ball
def track_ball(video, tracked_points, mask_lower, mask_upper, show_video=False):
	while True:
		# grab the current frame
		frame = video.read()
		# handle the frame from VideoCapture or VideoStream
		frame = frame[1] if video else frame
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
		mask = cv2.inRange(hsv, mask_lower, mask_upper)
		mask = cv2.erode(mask, None, iterations=3)
		mask = cv2.dilate(mask, None, iterations=3)
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
		tracked_points.append(center)

			# loop over the set of tracked points
		for i in range(1, len(tracked_points)):
			# if either of the tracked points are None, ignore them
			if tracked_points[i - 1] is None or tracked_points[i] is None:
				continue
			# otherwise, compute the thickness of the line and
			# draw the connecting lines
			thickness = int(np.sqrt(512 / float(i + 1)) * 2.5)
			cv2.line(frame, tracked_points[i - 1], tracked_points[i], (0, 0, 255), thickness)
		if show_video:
			# show the frame to our screen
			cv2.imshow("Frame", frame)
			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):
				break
	return tracked_points

tracked_pts = track_ball(cap, tracked_pts, yellowLower, yellowUpper)
# cap.release()
# cv2.destroyAllWindows()

# Plot the trajectory
def interpolate_nones(A: np.ndarray):
	ok = ~np.isnan(A)
	xp = ok.ravel().nonzero()[0]
	fp = A[~np.isnan(A)]
	x  = np.isnan(A).ravel().nonzero()[0]
	A[np.isnan(A)] = np.interp(x, xp, fp)
	return A

x_coords = interpolate_nones(np.array([pt[0] if pt is not None else np.nan for pt in tracked_pts]))
y_coords = interpolate_nones(np.array([pt[1] if pt is not None else np.nan for pt in tracked_pts]))

def plot_trajectory(x1, y1, x2, y2, title1, title2):
	if x2 is None or y2 is None:
		plt.figure()
		plt.plot(x1, y1, label=title1)
		plt.gca().invert_yaxis()
		plt.xlabel('x-coordinate (pixels)')
		plt.ylabel('y-coordinate (pixels)')
		plt.legend()
		plt.show()
	else:
		plt.figure()
		plt.plot(x1, y1, label=title1)
		plt.plot(x2, y2, label=title2)
		plt.gca().invert_yaxis()
		plt.xlabel('x-coordinate (pixels)')
		plt.ylabel('y-coordinate (pixels)')
		plt.legend()
		plt.show()

# Find the index of the maximum y-coordinate
max_index = np.argmax(y_coords)
print(f"The maximum y-coordinate occurs at frame {max_index}")

# Split x_coords and y_coords into incoming and outgoing arrays
incoming_x = x_coords[:max_index+1]
incoming_y = y_coords[:max_index+1]
outgoing_x = x_coords[max_index:]
outgoing_y = y_coords[max_index:]

def fit_line(x, y, order=1):
	coeffs = np.polyfit(x, y, order)
	fitted = np.polyval(coeffs, x)
	return fitted

def fit_parabola(x, y, order=2):
	coeffs = np.polyfit(x, y, order)
	fitted = np.polyval(coeffs, x)
	return fitted

fitted_incoming_y = fit_parabola(incoming_x, incoming_y)
fitted_outgoing_y = fit_parabola(outgoing_x, outgoing_y)

def get_angle(x, y):
	# Calculate the slope of the line between the first and last points
	slope = (y[-1] - y[0]) / (x[-1] - x[0])
	# Calculate the angle in degrees
	angle = np.arctan(slope) * 180 / np.pi
	return angle

incoming_angle = np.abs(np.round(get_angle(incoming_x, incoming_y), 2))
outgoing_angle = np.abs(np.round(get_angle(outgoing_x, outgoing_y), 2))

print(f"angle of incoming trajectory: {incoming_angle} degrees")
print(f"angle of outgoing trajectory: {outgoing_angle} degrees")

incoming_vx = np.trim_zeros(np.diff(incoming_x) * frame_rate / PIXELS_PER_METER)
incoming_vy = np.trim_zeros(np.diff(incoming_y) * frame_rate / PIXELS_PER_METER)
outgoing_vx = np.trim_zeros(np.diff(outgoing_x) * frame_rate / PIXELS_PER_METER)
outgoing_vy = np.trim_zeros(np.diff(outgoing_y) * frame_rate / PIXELS_PER_METER)

def plot_speeds(x, y, title1, title2):
	plt.figure()
	plt.subplot(2, 1, 1)
	plt.plot(x)
	plt.xlabel('Frame')
	plt.ylabel('Speed (m/s)')
	plt.title(title1)
	plt.subplot(2, 1, 2)
	plt.plot(y)
	plt.xlabel('Frame')
	plt.ylabel('Speed (m/s)')
	plt.title(title2)
	plt.tight_layout()
	plt.show()

def fit_curve(velocity_x, velocity_y, order):
	# cubic fit
	coeffs_vx = np.polyfit(np.linspace(0, len(velocity_x), len(velocity_x)), velocity_x, order)
	coeffs_vy = np.polyfit(np.linspace(0, len(velocity_y), len(velocity_y)), velocity_y, order)
	# Calculate the fitted speeds
	fitted_vx = np.polyval(coeffs_vx, np.linspace(0, len(velocity_x), len(velocity_x)))
	fitted_vy = np.polyval(coeffs_vy, np.linspace(0, len(velocity_y), len(velocity_y)))
	return fitted_vx, fitted_vy

fitted_incoming_vx, fitted_incoming_vy = fit_curve(incoming_vx, incoming_vy, 5)
fitted_outgoing_vx, fitted_outgoing_vy = fit_curve(outgoing_vx, outgoing_vy, 1)

# Skip to the frame at max_index
cap.set(cv2.CAP_PROP_POS_FRAMES, max_index)

# Read the frame at max_index
ret, impact_frame = cap.read()
# # Check if the frame was successfully read
# if ret:
#     # Display the frame
#     cv2.imshow('Frame at max_index', impact_frame)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# Release the VideoCapture
cap.release()

# Initialize the list of points
ball_distance2edge_points = []

# Mouse callback function
def click_event(event, x, y, flags, params):
    # If the left mouse button was clicked, record the (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        ball_distance2edge_points.append((x, y))

        # Draw a circle where the user clicked
        cv2.circle(impact_frame, (x, y), 5, (0, 255, 0), -1)

        # If two points have been clicked, draw a line between them
        if len(ball_distance2edge_points) == 2:
            cv2.line(impact_frame, ball_distance2edge_points[0], ball_distance2edge_points[1], (0, 255, 0), 2)

        # Display the image
        cv2.imshow('image', impact_frame)

# Calculate the distance between two points
def get_ball_distance_to_edge(impact_frame, points, return_type='x'):
	# Display the image and set the mouse callback function
	cv2.imshow('image', impact_frame)
	cv2.setMouseCallback('image', click_event)

	# Wait for a key press and then close the windows
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# If two points were clicked, calculate and print the distance between them
	if len(points) == 2:
		distance_to_edge = np.sqrt((points[1][0] - points[0][0])**2 + (points[1][1] - points[0][1])**2)
		x_distance = np.abs(points[1][0] - points[0][0])
		y_distance = np.abs(points[1][1] - points[0][1])
		if return_type == 'x':
			return x_distance
		elif return_type == 'y':
			return y_distance
		else:
			return distance_to_edge

ball_x_distance = np.round(get_ball_distance_to_edge(impact_frame, ball_distance2edge_points) / PIXELS_PER_METER * 100, 2)
print(f"Distance from ball to edge: {ball_x_distance} centimeters")

# print the fitted speeds
print(f'Incoming fitted speed in x: {np.round(fitted_incoming_vx, 2)} m/s')

if SHOW_PLOTS:
	# plot the trajectory
	plot_trajectory(x_coords, y_coords, None, None, 'Ball trajectory', None)
	# plot the incoming and outoging trajectories
	plot_trajectory(incoming_x, incoming_y, outgoing_x, outgoing_y, 'Incoming trajectory', 'Outgoing trajectory')
	# plot the incoming and outgoing fitted trajectories
	plot_trajectory(incoming_x, fitted_incoming_y, outgoing_x, fitted_outgoing_y, 'Incoming fit', 'Outgoing fit')
	# Plot the speeds in x and y as separate sub plots
	plot_speeds(incoming_vx, incoming_vy, 'Incoming speed in x', 'Incoming speed in y')
	plot_speeds(outgoing_vx, outgoing_vy, 'Outgoing speed in x', 'Outgoing speed in y')
	# Plot the fitted speeds in x and y as separate sub plots
	plot_speeds(fitted_incoming_vx, fitted_incoming_vy, 'Incoming fitted speed in x', 'Incoming fitted speed in y')
	plot_speeds(fitted_outgoing_vx, fitted_outgoing_vy, 'Outgoing fitted speed in x', 'Outgoing fitted speed in y')

if WRITE_DATA_TO_CSV:
	# Assuming fitted_incoming_vx, fitted_incoming_vy, ball_x_distance, incoming_angle, and outgoing_angle are defined
	data = {
		'fitted_incoming_vx': [fitted_incoming_vx],
		'fitted_incoming_vy': [fitted_incoming_vy],
		'ball_x_distance': [ball_x_distance],
		'incoming_angle': [incoming_angle],
		'outgoing_angle': [outgoing_angle]
	}

	# Create a DataFrame from the data
	df = pd.DataFrame(data)

	# Write the DataFrame to a CSV file
	df.to_csv('output.csv', index=False)