# Traffic — Traffic Sign Classifier (CNN)

Train a small convolutional neural network to classify traffic sign images
from a directory-structured dataset (one folder per category).

**Overview**
- `traffic.py` loads image data, builds a Keras CNN, trains and evaluates it,
  and can save the trained model.

**Requirements**
- Python 3.8+
- `opencv-python`, `tensorflow` (or `keras`), `scikit-learn`, `numpy`

**Quick Start**
```bash
cd traffic
python traffic.py gtsrb
python traffic.py gtsrb model.keras
```
