import cv2

# Open the video file
cap = cv2.VideoCapture('high_n.MOV')
cap.set(cv2.CAP_PROP_POS_MSEC, 500)      # Go to the 0.5 sec. position
success, image = cap.read()

# Resize the image
image_resized = cv2.resize(image, (960, 540))  # Change to desired dimensions

# Select the ROI on the resized image
r = cv2.selectROI(image_resized)

# Adjust ROI to original image scale
r = (r[0]*image.shape[1]/960, r[1]*image.shape[0]/540, r[2]*image.shape[1]/960, r[3]*image.shape[0]/540)

# Print ROI
print(r)

# Crop the image
crop = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
print(crop.shape)
# print the cropped coordinates
print(f"Top left Y and y distance: ({int(r[1])}, {int(r[1]+r[3])})")
print(f"Top left X and the x distance: ({int(r[0])}, {int(r[0]+r[2])})")


# Display the cropped image
cv2.imshow("Image", crop)
cv2.waitKey(0)