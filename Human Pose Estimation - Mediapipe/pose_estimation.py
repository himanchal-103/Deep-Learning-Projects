# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp


# Initialize Mediapipe drawing and pose modules
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)



# # Pose landmark detection for a static image
# # Read the input image
# img = cv2.imread('action_recognition/13.jpg')

# # Process the image to detect pose landmarks 
# results = pose.process(img)  

# # Draw landmarks on the original image
# mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
#                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
#                               ) 
# # Create a blank white image to draw extracted keypoints
# h, w, c = img.shape
# opImg = np.zeros([h, w, c])
# opImg.fill(255)
# mp_drawing.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
#                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4),
#                               mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=4)
#                               )

# # Display the original image with landmarks and the extracted keypoints image
# cv2.imshow('Original image', img)
# cv2.imshow('Extracted Keypoints', opImg)
 
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# Pose landmarks detection on video
cap = cv2.VideoCapture('action_recognition/barbell_deadlift.mp4')
# cap = cv2.VideoCapture('action_recognition/video.mp4')

while True:
    ret, frame = cap.read() # Read a frame from the video
    if not ret:
        print('Image not available')  # Break the loop if there is no frame
        break
    
    img = cv2.resize(frame, (600, 400)) # resizing frame if frame size if big
    results = pose.process(img)  # Process the frame to detect pose landmarks

    # Draw landmarks on original frame
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
                              ) 
    
    # Create a blank white image to draw extracted keypoints
    h, w, c = img.shape
    opImg = np.zeros([h, w, c])
    opImg.fill(255)
    mp_drawing.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)
                              )
    
    # Display the original frame with landmarks and the extracted keypoints frame
    cv2.imshow('Original video', img)
    cv2.imshow('Extracted Keypoints', opImg)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop when 'q' key is pressed
        break

cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows
