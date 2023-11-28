
import cv2

# Open the video file
cap = cv2.VideoCapture('ball-slo.mp4')

# Get the frame count and frame rate
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_rate = cap.get(cv2.CAP_PROP_FPS)

# Calculate the duration of the video in seconds
duration = frame_count / frame_rate

print(f'Frame rate: {frame_rate:.2f} fps')
print(f'Duration: {duration:.2f} seconds')

# Release the video capture object
cap.release()
