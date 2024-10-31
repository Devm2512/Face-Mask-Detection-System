# Face Mask Detection Project

This project detects faces from images and videos, then classifies them as "with mask" or "without mask" using a pre-trained model. It is divided into three components:

1. **Backend (back.py)**: Handles face detection and mask prediction using OpenCV and a deep learning model.
2. **Frontend (frontend.py)**: Manages the user interface and interactions (details to be provided by the frontend.py code).
3. **Main (main.py)**: Serves as the central script to run and integrate the backend and frontend functionalities (details to be provided by main.py code).

## Table of Contents
- [Project Overview](#project-overview)
- [Backend (back.py)](#backend-backpy)
- [Frontend (frontend.py)](#frontend-frontendpy)
- [Main (main.py)](#main-mainpy)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to detect whether individuals in a video feed or image are wearing face masks. It utilizes a face detection model and a pre-trained mask classification model built with TensorFlow/Keras.

### Dataset
The Dataset contains 2 folders:
- Train
- Test
Each folder further contains 2 folders.
- With mask
- Without mask

### Backend (back.py)
The `back.py` script is responsible for:
- Loading an image or video file.
- Detecting faces using a pre-trained face detection model (Haar Cascade).
- Predicting whether the detected faces are wearing masks using a pre-trained Keras model (`mask.h5`).
- Drawing bounding boxes around faces: red for "without mask" and green for "with mask".

### Frontend (frontend.py)
This script manages the user interface and interactions. It allows users to:
- Choose between uploading an image or selecting a video file for analysis.
- View the real-time results of mask detection.

### Main (main.py)
The `main.py` script integrates both backend and frontend, running the complete application, coordinating between the user input and the backend processing, and displaying the results.

## Installation

Follow the steps below to set up and run the project:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
