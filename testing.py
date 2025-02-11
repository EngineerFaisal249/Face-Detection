import cv2
from ultralytics import YOLO

# Load the trained face detection model
model = YOLO(r'yolov8n-face.pt')

# Path to the video file
video_path = r'vid1.mp4'

# Start video capture from the video file
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define video writers for different formats with matching frame size
output_mp4 = cv2.VideoWriter('output_face_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
output_avi = cv2.VideoWriter('output_face_video.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
output_mkv = cv2.VideoWriter('output_face_video.mkv', cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or unable to read the frame.")
        break

    # Perform detection
    results = model(frame)
    detections = results[0].boxes  # Get detected boxes

    # Draw bounding boxes on the frame
    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 4)  # Yellow color with thickness 4

        # Optionally, add labels
        label = "Person"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # Display the annotated frame
    cv2.imshow('Face Re-identification', frame)

    # Write the annotated frame to all output videos
    output_mp4.write(frame)
    output_avi.write(frame)
    output_mkv.write(frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
output_mp4.release()
output_avi.release()
output_mkv.release()
cv2.destroyAllWindows()
