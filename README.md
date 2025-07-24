# 🍽️ Food Image Classifier

This project trains a deep learning model to classify food images into one of 101 categories using the Food-101 dataset and ResNet18.

---

## 📁 Project Structure
```
food-image-classifier/
├── data/
│   └── food-101/
│       ├── images/               # Contains all original images
│       ├── images_split/         # Will contain train/ and test/ after splitting
│       ├── meta/                 # Contains train.txt and test.txt split info
├── models/
│   └── (empty initially; model will be saved here)
├── split_dataset.py              # Script to create train/test folders
├── train.py
├── predict.py *(to be added)*
├── app.py *(to be added)*
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository
```bash
git clone <your-github-repo-url>
cd food-image-classifier
```

### ✅ 2. Set Up a Virtual Environment with Python 3.10 or 3.11
Make sure you are **not using Python 3.13+**, as PyTorch doesn't support it yet.

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### ✅ 3. Install Dependencies
```bash
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install streamlit
```

Or use the `requirements.txt`:
```txt
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
streamlit==1.35.0
```

```bash
pip install -r requirements.txt
```

---

## 📦 Download the Dataset

### Run the following:
```bash
cd data/
wget https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar -xvzf food-101.tar.gz
rm food-101.tar.gz
cd ..
```

Make sure your dataset is structured like:
```
data/food-101/images/train/apple_pie/image1.jpg
```

You can verify category folders:
```bash
ls data/food-101/images/train | wc -l  # should show 101
```

If the `train/` and `test/` folders do not exist, run the provided helper script:

### 🧩 Split the Dataset into Train/Test Folders
```bash
python split_dataset.py
```
This will read from `meta/train.txt` and `meta/test.txt` and create:
```
data/food-101/images_split/train/
                         /test/
```

Then update `DATA_DIR` in `train.py` to point to `images_split`:
```python
DATA_DIR = "data/food-101/images_split"
---

## 🚀 Train the Model
```bash
python train.py
```

This will train a ResNet18 model and save it to:
```
models/food101_resnet.pth
```

During training, the script now also evaluates **validation accuracy** at the end of each epoch and prints it to the console.

---

## 🧪 Coming Next
- `predict.py` for inference
- `app.py` with Streamlit UI
- GitHub-ready visual demo

Stay tuned!
