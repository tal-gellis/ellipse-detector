import os
import json
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# הגדרות
DATA_DIR = 'data'          # תקיית הקבצים
IMG_SIZE = (64, 64)        # גודל אחיד לכל התמונות

def load_image(image_path, size=IMG_SIZE):
    """טוען תמונה וממיר אותה למטריצה מנורמלת בגודל אחיד"""
    img = Image.open(image_path).convert('L')  # 'L' = grayscale
    img = img.resize(size)
    img_array = np.array(img, dtype=np.float32) / 255.0  # נרמול
    return img_array  # גודל: (64, 64)

def load_label(json_path):
    """טוען את הפרמטרים מתוך קובץ JSON ומחזיר וקטור של 6 ערכים מנורמלים"""
    with open(json_path, 'r') as f:
        data = json.load(f)

    is_ellipse = 1.0 if data.get('isEllipse', False) else 0.0

    if is_ellipse:
        x, y = data['centerCoord']
        major, minor = data['radLength']
        angle = data['rotAngle']
        
        # Normalize coordinates to [0, 1] relative to image size (assuming original coords are for 64x64)
        x_norm = x / 64.0
        y_norm = y / 64.0
        
        # Normalize radii to [0, 1] relative to image size
        major_norm = major / 32.0  # Max radius is half the image size
        minor_norm = minor / 32.0
        
        # Normalize angle to [0, 1] from [0, 2π]
        angle_norm = angle / (2 * np.pi)
        
    else:
        x_norm, y_norm, major_norm, minor_norm, angle_norm = 0.0, 0.0, 0.0, 0.0, 0.0

    return np.array([is_ellipse, x_norm, y_norm, major_norm, minor_norm, angle_norm], dtype=np.float32)

def load_dataset(data_dir, split='all', test_size=0.2, random_state=42):
    """
    מחזיר רשימות של תמונות ותגיות
    Args:
        data_dir: תקיית הנתונים
        split: 'all', 'train', or 'test'
        test_size: אחוז הנתונים לבדיקה (0.2 = 20%)
        random_state: זרע לשחזוריות
    """
    images = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith('.png'):
            base = file[:-4]  # מסיר '.png'
            img_path = os.path.join(data_dir, f'{base}.png')
            json_path = os.path.join(data_dir, f'{base}.json')

            if not os.path.exists(json_path):
                continue  # מדלג אם אין קובץ JSON תואם

            img = load_image(img_path)
            label = load_label(json_path)

            images.append(img)
            labels.append(label)

    # המרה למערכים
    X = np.expand_dims(np.array(images), axis=-1)  # shape: (N, 64, 64, 1)
    y = np.array(labels)                           # shape: (N, 6)
    
    if split == 'all':
        return X, y
    
    # פיצול לאימון ובדיקה
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y[:, 0]
    )
    
    if split == 'train':
        print(f"📚 Training set: {len(X_train)} samples")
        return X_train, y_train
    elif split == 'test':
        print(f"🧪 Test set: {len(X_test)} samples")
        return X_test, y_test
    else:
        raise ValueError("split must be 'all', 'train', or 'test'")