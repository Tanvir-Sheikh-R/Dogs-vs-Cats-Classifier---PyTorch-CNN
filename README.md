# Dogs vs Cats Classifier 🐶🐱

A custom Convolutional Neural Network (CNN) built from scratch using PyTorch to classify images of dogs and cats. Achieved **~93% validation accuracy** on the Kaggle Dogs vs Cats dataset.

---

## 📁 Dataset

The dataset used in this project is from the Kaggle Dogs vs Cats competition.

🔗 [Download Dataset from Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data)

The dataset contains:
- **25,000** labeled training images (cats and dogs)
- **12,500** unlabeled test images

---

## 🧠 Model Architecture

A custom CNN with the following structure:

- **3 Convolutional Blocks**, each with:
  - 2x Conv2D + BatchNorm + ReLU
  - MaxPooling
  - Dropout (0.2 → 0.3 → 0.4)
- **Fully Connected Layers**:
  - LazyLinear(512) + BatchNorm + ReLU + Dropout
  - Linear(512 → 1) for binary output

---

## ⚙️ Training Details

| Parameter | Value |
|---|---|
| Input Size | 112 x 112 |
| Batch Size | 64 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |
| Epochs | 25 |
| Val Accuracy | ~93% |

---

## 🔄 Data Augmentation

**Training:**
- Random Horizontal Flip
- Random Rotation (±10°)
- Random Crop (112x112)
- Color Jitter (brightness & contrast)
- Normalize (mean=0.5, std=0.5)

**Validation/Test:**
- Center Crop (112x112)
- Normalize (mean=0.5, std=0.5)

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/dogs-vs-cats-classifier.git
cd dogs-vs-cats-classifier
```

### 2. Install dependencies
```bash
pip install torch torchvision pillow pandas numpy matplotlib
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data) and place it in the `data/` folder:
```
data/
├── train/        ← original training images
├── train1/       ← sorted into cat/ and dog/ subfolders
│   ├── cat/
│   └── dog/
└── test1/        ← test images
```

### 4. Train the model
Open and run `Dogs_vs_Cats.ipynb` in Jupyter Notebook.

### 5. Run inference
```python
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

---

## 📦 Requirements

```
torch
torchvision
pillow
pandas
numpy
matplotlib
```

---

## 📊 Results

| Epoch | Loss | Val Accuracy |
|---|---|---|
| 16 | 0.2200 | 91.34% |
| 20 | 0.2033 | 91.96% |
| 25 | 0.1743 | 92.75% |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
