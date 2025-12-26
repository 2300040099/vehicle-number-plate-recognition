# Vehicle Number Plate Recognition System

This project detects vehicle number plates from video input
and extracts the plate text using optical character recognition (OCR).

## Technologies Used
- Python
- OpenCV
- Haar Cascade Classifier
- Tesseract OCR

## Files
- plate_detection.py  
  Detects number plates from video frames and extracts text using OCR.

- haarcascade_russian_plate_number.xml  
  Pretrained Haar Cascade used for number plate detection.

## Working Flow
1. Video frames are captured from a video file or webcam.
2. Number plates are detected using Haar Cascade.
3. Detected plate regions are preprocessed.
4. Tesseract OCR is used to extract the plate number text.
5. Output is displayed and saved to a video file.

## Input
- Video file (`video.mp4`) or live camera feed

## Output
- Bounding box around detected number plates
- Extracted plate number displayed on frame
- Output video saved as `output.avi`

## Applications
- Traffic monitoring systems
- Automated toll collection
- Smart parking solutions

## Limitations
- Uses a pretrained Haar Cascade for specific plate formats
- OCR accuracy depends on lighting and plate quality

## Future Enhancements
- Support for multiple country number plates
- Deep learning based plate detection
- Improved OCR accuracy
