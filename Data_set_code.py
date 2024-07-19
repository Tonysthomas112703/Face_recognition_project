import cv2
import os

# Open the video file
v_cap = cv2.VideoCapture('C:/Users/ROSEMILK/Downloads/aswathy-dataset/devika.mp4')
v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create a directory to save the frames
save_dir = "C:/Users/ROSEMILK/OneDrive/Documents/asw/dev"
os.makedirs(save_dir, exist_ok=True)

# Loop through the video frames
for i in range(v_len):
    success, frame = v_cap.read()
    if not success:
        continue
    
    # Save the frame as an image
    cv2.imwrite(f'{save_dir}/frame_{i}.jpg', frame)

# Release the video capture object
v_cap.release()