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
│       └── meta/                 # Contains train.txt and test.txt split info
├── models/
│   ├── checkpoints/              # Contains checkpoint subdirs by config
│   └── final/                    # Stores final trained models
├── logs/                         # Contains training logs and summaries
├── output/                       # Stores predictions from predict.py
├── analysis/                     # Stores evaluation reports and plots
├── utils/
│   ├── data_loader.py
│   ├── logger.py
│   ├── model_builder.py
│   ├── trainer.py
│   └── validator.py
├── tests/
│   ├── __init__.py
│   ├── test_config.py            # Unit testing Configs
│   ├── test_utils.py             # Unit testing Utils
│   ├── test_data_loader.py       # Unit testing for data loading
│   ├── test_logger.py            # Unit testing for logging logic
│   └── test_predict.py           # Unit testing for predictions
├── config.py                     # CLI config parser
├── split_dataset.py              # Script to create train/test folders
├── train.py                      # Main training entrypoint
├── predict.py                    # Inference script
├── analysis/
│   └── prediction_analysis.py    # Model evaluation & visualization
├── app.py *(to be added)*       
├── requirements.txt
└── README.md
```

---

## 🛠️ Setup Instructions

### ✅ 1. Clone the Repository
```bash
git clone https://github.com/wafaahs/food-classifier.git
cd food-classifier
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
numpy==1.26.4
pillow==10.3.0
tqdm==4.66.4
streamlit==1.35.0
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.5.0
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
```

---

## 🚀 Train the Model

```bash
python train.py --epochs 10 --batch_size 64 --lr 0.0005 --optimizer sgd
```

### 🔧 Command-Line Options

You can pass the following optional arguments to customize training:

- `--epochs`: number of training epochs (default: 5)
- `--batch_size`: mini-batch size (default: 32)
- `--lr`: learning rate (default: 0.001)
- `--optimizer`: optimization algorithm (`adam` or `sgd`, default: `adam`)

> If omitted, the script will fall back to defaults defined in `config.py`.

- Resumes from last checkpoint if available
- Logs progress to console and `logs/`
- Saves final model to `models/final/`
- Outputs epoch summaries in CSV format

---

### 🗓️ Output Structure

- Logs: `logs/training_<timestamp>.log`
- Summary CSV: `logs/summary_<timestamp>.csv`
- Checkpoints: `models/checkpoints/e{epochs}_b{batch}_opt_lr.../`
- Final model: `models/final/e10_b64_sgd_lr0p0005_resnet.pth`

---

## 🧠 Inference: Make Predictions

Once training is complete, use `predict.py` to run inference on the test dataset.

```bash
python predict.py \
  --model models/final/e10_b64_adam_lr0p001_resnet.pth \
  --data_dir data/food-101/images_split/test \
  --output_dir output/
```

### 🔧 Command-Line Options
- `--model`: path to the trained .pth model
- `--data_dir`: directory containing test images
- `--output_dir`: base folder for saving predictions

This will:

- Predict all images in the test directory
- Create a subdirectory in `output/` named after the model config
- Save `predictions.csv` with the following format:

```csv
file_path, prediction, ground_truth
data/food-101/images_split/test/ramen/123.jpg, ramen, ramen
data/food-101/images_split/test/apple_pie/456.jpg, cheesecake, apple_pie
```

> ⚠️ Ensure the ground-truth class is inferable from the directory name for evaluation.

---

## 📊 Analyze Predictions

Use `prediction_analysis.py` to evaluate your model’s performance:

```bash
python analysis/prediction_analysis.py --subdir e10_b64_adam_lr0p001
```
### 🔧 Command-Line Options
- `--subdir`: name of the model subdirectory (e.g. `e10_b64_adam_lr0p001`) inside `output/`


This will automatically:

- Load `output/e10_b64_adam_lr0p001/predictions.csv`
- Create results in: `analysis/e10_b64_adam_lr0p001/`
- Generate:
  - `classification_report.csv`
  - `summary.txt` (with worst 5 classes)
  - `confusion_matrix.png` (multi-class)
  - `class_metrics_bar_chart.png` (precision/recall/F1)
  - Per-class 2×2 confusion matrices labeled with TP/FN/FP/TN and F1

---

## 🤪 Coming Next

- `app.py` with Streamlit UI
- GitHub-ready visual demo
- End-to-end notebook

Stay tuned!
