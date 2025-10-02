
# Ellipse Detection with Deep Learning

This project was developed as part of a **candidate assessment task** for a hiring process. It uses a **convolutional neural network (CNN) implemented with Keras** to detect ellipses in images and estimate their geometric parameters. The development was assisted by the **Cursor AI coding tool**.

## Project Overview

The model can:

* Detect whether an ellipse exists in an image
* Locate the center coordinates (x, y)
* Estimate the major and minor axis lengths
* Calculate the rotation angle of the ellipse

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train the model:

```bash
python main.py --mode train
```

3. Evaluate performance:

```bash
python main.py --mode eval
```

## Technical Details

* Input: 64x64 grayscale images
* Output: Classification (ellipse present or not) + 5 regression parameters
* CNN architecture optimized for small size and high accuracy

## Disclaimer

This project was completed as part of a **hiring assessment** and development was assisted by **Cursor AI**.

