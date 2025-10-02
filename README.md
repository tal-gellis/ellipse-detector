# 🔍 Ellipse Detection with Deep Learning

A deep learning project that detects ellipses in noisy images and estimates their parameters using a convolutional neural network.

## 📝 What This Project Does

This AI model can:
1. **Detect** whether an ellipse exists in an image
2. **Locate** the center coordinates (x, y) of the ellipse
3. **Measure** the major and minor axis lengths
4. **Calculate** the rotation angle of the ellipse

### Example Input/Output:
- **Input**: 64x64 grayscale image with noise
- **Output**: 
  - ✅ "Ellipse detected: YES"
  - 📍 Center: (32, 28)
  - 📏 Major axis: 25 pixels, Minor axis: 15 pixels
  - 🔄 Rotation: 45 degrees

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python main.py --mode train
```

### 3. Evaluate Performance
```bash
python main.py --mode eval
```

## 📊 Performance Results

**Tested on 820 unseen images:**
- 🎯 **Classification Accuracy**: 90.4%
- 🔍 **Precision**: 99.7% (very few false positives)
- 📈 **Recall**: 88.9% (catches most ellipses)
- 📐 **Parameter Estimation**: Average errors:
  - Center coordinates: ~20-23% error
  - Axis lengths: ~17% error  
  - Rotation angle: ~29% error

## 🧠 Model Architecture

**Optimized CNN with 555,286 parameters (under 750K limit):**

```
Input (64x64x1) → Conv(8→16→32→64) → Flatten → FC(128→48) → Output (6 values)
```

**Layer Details:**
- 4 Convolutional layers with ReLU activation and MaxPooling
- 2 Fully connected layers with dropout (0.3) for regularization
- Multi-task output: 1 classification + 5 regression parameters

**Output Format:**
1. `isEllipse`: 0 or 1 (classification)
2. `x_coord`: Center X coordinate (normalized)
3. `y_coord`: Center Y coordinate (normalized)  
4. `major_axis`: Major radius length (normalized)
5. `minor_axis`: Minor radius length (normalized)
6. `rotation`: Angle in radians (normalized)

## 📁 Project Structure

```
ellipse.ai/
├── main.py              # Main script to run training/evaluation
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── src/
│   ├── config.py       # Configuration settings
│   ├── model.py        # CNN model architecture
│   ├── data_loader.py  # Data loading and preprocessing
│   ├── train.py        # Training script
│   └── evaluate.py     # Evaluation script
├── data/               # Training images and labels
│   ├── Image0001.png   # Sample images
│   ├── Image0001.json  # Ground truth labels
│   └── ...
└── outputs/            # Generated files
    ├── model.pt        # Trained model
    ├── loss_plot.png   # Training loss graph
    └── metrics.png     # Evaluation results
```

## 🔧 Technical Details

### Data Format
- **Images**: 64x64 grayscale PNG files
- **Labels**: JSON files with ground truth parameters
- **Split**: 80% training (3,276 images) / 20% testing (820 images)

### Training Configuration
- **Epochs**: 10
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss**: Binary Cross-Entropy + MSE (multi-task learning)

### Data Preprocessing
- Images normalized to [0, 1]
- Coordinates normalized by image size
- Radii normalized by max possible radius
- Angles normalized to [0, 1] range

## 📋 Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Matplotlib
- Pillow (PIL)
- scikit-learn

## 🎯 Use Cases

This model can be applied to:
- **Medical imaging**: Detecting circular/elliptical structures
- **Quality control**: Finding defects in manufactured parts
- **Computer vision**: Object detection and shape analysis
- **Astronomy**: Detecting celestial objects
- **Industrial inspection**: Automated quality assessment

## 📈 How It Works

1. **Input Processing**: Image is resized to 64x64 and normalized
2. **Feature Extraction**: CNN layers detect visual patterns
3. **Multi-task Output**: Model simultaneously predicts:
   - Classification (ellipse vs no ellipse)
   - Regression (5 geometric parameters)
4. **Post-processing**: Predictions are denormalized to original scale

## 🔍 Model Performance Details

The model uses a **multi-task learning** approach:
- **Classification Head**: Detects presence of ellipse
- **Regression Head**: Estimates ellipse parameters (only when ellipse exists)

**Loss Function**: `Total Loss = Classification Loss + Regression Loss`

## 🛠️ Customization

To modify the model:
- **Change image size**: Update `IMG_SIZE` in `config.py`
- **Adjust model complexity**: Modify layers in `model.py`
- **Tune training**: Update hyperparameters in `config.py`
- **Add data augmentation**: Extend `data_loader.py`

## 📞 Notes

- The model is designed to be simple yet effective
- Training takes ~2-3 minutes on modern hardware
- Results are reproducible with fixed random seed
- Model automatically handles GPU if available

---

*Built with PyTorch • Optimized for accuracy and speed • Ready for production use*
