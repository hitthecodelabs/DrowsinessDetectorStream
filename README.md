# DrowsinessDetectorStream

## Description
DrowsinessDetectorStream is a real-time drowsiness detection system using a webcam feed. It utilizes OpenCV for image processing, Dlib for facial landmark detection, and Pygame for alarm notifications. The application is built on Flask for easy web streaming of the video feed.

## Features
- Real-time drowsiness detection using webcam feed.
- Facial landmark analysis to detect eyes, mouth, and nose positions.
- Eye Aspect Ratio (EAR) algorithm for detecting closed eyes.
- Audio alerts for detected drowsiness.
- Web interface for streaming the video feed.

## Installation

### Prerequisites
- Python 3.x
- OpenCV
- Dlib
- Pygame
- Pillow
- Flask

### Setup
1. Clone the repository:

```bash
git clone https://github.com/[your-username]/DrowsinessDetectorStream.git
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
To start the application, run the following command in your terminal:
```bash
python app_cv2.py
```

Open a web browser and navigate to `http://localhost:5000` to view the streaming video feed.

## Contribution
Contributions to DrowsinessDetectorStream are welcome. Please ensure to follow best practices and code standards when contributing.

## License
[MIT License](LICENSE)

## Acknowledgments
- Haar Cascades for face and eye detection.
- Dlib's facial landmark predictor.
- Pygame for audio playback.
- Flask for web framework capabilities.

## Contact
For any inquiries, please contact [e-mail](mailto:jpaul@hitthecodelabs.com).
