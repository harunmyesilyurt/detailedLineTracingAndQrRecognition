import cv2
import numpy as np

# Start the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # If reading from the camera was successful, perform the operations
    if ret:
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Otsu Thresholding
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Define a region of interest to detect the gray square (e.g., the right side of the frame)
        roi = gray[:, gray.shape[1] // 2:]  # Right half of the frame
        _, roi_thresholded = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find contours to detect the gray square
        contours, _ = cv2.findContours(roi_thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # If the contour is above a certain area (to define the size of the gray square)
            if 500 < area < 5000:  # Adjust these values based on the size of the gray square
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x + gray.shape[1] // 2, y), (x + w + gray.shape[1] // 2, y + h), (0, 255, 0), 2)

        # Display the results
        cv2.imshow('Original', frame)
        cv2.imshow('Thresholded', thresholded)
        cv2.imshow('ROI Thresholded', roi_thresholded)

    # Break the loop and exit when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything
cap.release()
cv2.destroyAllWindows()
