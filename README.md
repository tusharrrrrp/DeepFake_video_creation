# Facial Replacement Video Generator

A tool for generating facial replacement videos (deepfakes) using MediaPipe and OpenCV.

## Overview

This project uses MediaPipe's FaceMesh to detect facial landmarks in both a source video and a target image. It then applies facial replacement techniques to map the facial expressions from the source video onto the target image, creating a realistic facial animation effect.

## Features

- Facial landmark detection using MediaPipe FaceMesh
- Delaunay triangulation for facial region mapping
- Affine transformation for realistic facial animation
- Compatible with Google Colab for cloud-based processing
- Supports various video formats and resolutions

## Installation

### Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/tusharrrrrp/DeepFake_video_creation.git
   cd DeepFake_video_creation
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Google Colab Usage

This project is optimized for use with Google Colab. To use it:

1. Upload the script to a new Colab notebook
2. Mount your Google Drive
3. Upload your source video and target image to your Google Drive
4. Adjust file paths in the script to match your Drive structure

## Usage

### Basic Usage

1. Prepare a source video (with facial movements) and a target image (face to be animated)
2. Set the file paths in the script:
   ```python
   source_video_path = 'path/to/your/source_video.mp4'
   target_image_path = 'path/to/your/target_image.png'
   output_video_path = 'path/to/save/output_video.avi'
   ```
3. Run the script:
   ```
   python main.py
   ```

### Google Colab Example

```python
from google.colab import drive
drive.mount('/content/drive')

# Set file paths
source_video_path = '/content/drive/My Drive/driver.mp4'
target_image_path = '/content/drive/My Drive/source.png'
output_video_path = '/content/drive/My Drive/deepfake_output.avi'

# Run the main script
%run main.py
```

## How It Works

1. **Facial Landmark Detection**: The script uses MediaPipe FaceMesh to detect 468 facial landmarks on both the source video frames and target image.

2. **Delaunay Triangulation**: The facial landmarks are used to create a triangular mesh on the face.

3. **Transformation**: For each triangle in the mesh, an affine transformation is calculated to map the target image triangle to the corresponding source video triangle.

4. **Warping and Blending**: Each triangle from the target image is warped according to the calculated transformation and blended into the output frame.

5. **Video Generation**: The process is repeated for each frame of the source video to generate the output video.

## Limitations

- Requires clear facial visibility in both source video and target image
- Best results are achieved with similar facial orientations
- Processing can be computationally intensive for high-resolution videos
- Currently designed for single-face processing

## Ethical Considerations

This tool is intended for educational, artistic, and entertainment purposes only. Users should:

- Obtain proper consent before using someone's likeness
- Clearly label synthetic media as such
- Avoid creating misleading or potentially harmful content
- Follow all applicable laws and regulations regarding synthetic media

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MediaPipe for the facial landmark detection technology
- OpenCV community for computer vision tools
