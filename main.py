from google.colab import drive
import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow

# Mount Google Drive
drive.mount('/content/drive')

# File paths
source_video_path = ''  # Source video with movements
target_image_path = ''  # Target image
output_video_path = '/content/deepfake_output01.avi'#avi extension

# Initialize MediaPipe FaceMesh detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to get landmarks using MediaPipe
def get_landmarks(image):
    """Detect facial landmarks in an image using MediaPipe FaceMesh."""
    # Convert image to RGB for MediaPipe
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        return [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks]
    return None

# Load the source video
source_video = cv2.VideoCapture(source_video_path)
fps = int(source_video.get(cv2.CAP_PROP_FPS))
frame_width = int(source_video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(source_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load and resize the target image to match source video resolution
target_image = cv2.imread(target_image_path)
target_image = cv2.resize(target_image, (frame_width, frame_height))

# Create VideoWriter to save output
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while True:
    ret, source_frame = source_video.read()
    if not ret:
        break

    # Get landmarks from the source video frame
    landmarks_source = get_landmarks(source_frame)
    if landmarks_source is None:
        continue

    # Get landmarks from the target image
    landmarks_target = get_landmarks(target_image)
    if landmarks_target is None:
        continue

    # Calculate Delaunay triangulation on target landmarks
    rect = (0, 0, frame_width, frame_height)
    subdiv = cv2.Subdiv2D(rect)

    # Insert each point as an integer tuple into the subdivider
    for point in landmarks_target:
        subdiv.insert((int(point[0]), int(point[1])))

    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    # Define transformation for each triangle
    output_frame = np.zeros_like(source_frame)
    for triangle in triangles:
        # Extract triangle points from target and source landmarks
        pts_target = [landmarks_target[i] for i in [np.where((landmarks_target == t).all(axis=1))[0][0]
                        for t in triangle.reshape(3, 2)]]
        pts_source = [landmarks_source[i] for i in [np.where((landmarks_target == t).all(axis=1))[0][0]
                        for t in triangle.reshape(3, 2)]]

        # Convert to NumPy arrays
        pts_target = np.array(pts_target, dtype=np.float32)
        pts_source = np.array(pts_source, dtype=np.float32)

        # Calculate affine transform
        matrix = cv2.getAffineTransform(pts_target, pts_source)

        # Warp the triangular region
        warped_triangle = cv2.warpAffine(target_image, matrix, (frame_width, frame_height))

        # Create masks to combine warped region
        mask = np.zeros_like(source_frame)
        cv2.fillConvexPoly(mask, np.int32(pts_source), (255, 255, 255))
        warped_triangle = cv2.bitwise_and(warped_triangle, mask)
        output_frame = cv2.add(output_frame, warped_triangle)

    # Write the output frame
    out.write(output_frame)

    # Display the frame (optional)
    cv2_imshow(output_frame)

# Release the video capture and writer
source_video.release()
out.release()

# Print the output location
print(f"Deepfake video saved at: {output_video_path}")
