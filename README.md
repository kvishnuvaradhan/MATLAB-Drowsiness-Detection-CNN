# MATLAB CNN Drowsiness Detection

Starter project for classifying open and closed eyes using a CNN in MATLAB, followed by a simple webcam demo.

## Files
- `train_drowsiness_cnn.m` — trains a CNN and saves `trainedDrowsinessNet.mat`.
- `realtime_drowsiness.m` — classifies webcam frames and displays an alert after 15 consecutive frames predicted as closed eyes.

## Dataset structure
Place a dataset folder beside these scripts:
```text
dataset/
├── Closed_Eyes/
└── Open_Eyes/
```
Add your labeled images to the correct folders. The dataset and trained `.mat` model are not included.

## Requirements
- MATLAB Deep Learning Toolbox
- Image Processing Toolbox
- MATLAB webcam support
- Computer Vision Toolbox recommended for `vision.CascadeObjectDetector`

## Run
1. Put images in `dataset/Closed_Eyes` and `dataset/Open_Eyes`.
2. Run `train_drowsiness_cnn.m`.
3. When training completes, run `realtime_drowsiness.m`.

## Important notes
This is a generated starter implementation based on the project details discussed; it is not recovered original code. You must train it with your dataset before running the webcam script. The detector may behave differently under different lighting and camera positions. This is an academic demo, not a safety-certified driver monitoring system.
