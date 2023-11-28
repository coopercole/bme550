import cv2
import numpy as np
import time

# Open the video file
cap = cv2.VideoCapture('high_n.MOV')

# Get the frame rate
frame_rate = cap.get(cv2.CAP_PROP_FPS)
print(f'Frame rate: {frame_rate}')

# Initialize the tracker
tracker = cv2.legacy.TrackerMOSSE_create()
_, frame = cap.read()
bbox = cv2.selectROI("Tracking",frame, False)
tracker.init(frame, bbox)

old_position = np.array([bbox[0], bbox[1]])
total_distance = 0

start_time = time.time()
# Process the video
while True:
    _, frame = cap.read()
    if not _:
        break

    _, bbox = tracker.update(frame)
    new_position = np.array([bbox[0], bbox[1]])

    # Calculate the distance travelled
    distance = np.linalg.norm(new_position - old_position)
    total_distance += distance

    old_position = new_position

end_time = time.time()

# Calculate the speed
duration = end_time - start_time
# duration1 = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # duration in seconds
print(f'Duration: {duration} seconds')
print(f'Total distance: {total_distance} pixels')
speed = total_distance / duration  # speed in pixels per second
print(f'Speed: {speed} pixels/second')
speed_in_meters_per_second = speed / 811.46
print(f'Speed: {speed_in_meters_per_second} m/s')

# Release the video capture object
cap.release()