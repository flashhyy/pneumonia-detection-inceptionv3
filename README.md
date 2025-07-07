
# Pneumonia Detection using Transfer Learning (InceptionV3)

## 📌 Objective
To classify chest X-ray images into **Pneumonia** or **Normal** using a pretrained deep learning model (InceptionV3) and evaluate its performance on the PneumoniaMNIST dataset.

---

## 📊 Dataset: PneumoniaMNIST
- Source: MedMNIST v2
- Size: 28×28 grayscale chest X-ray images
- Classes: `0` = Normal, `1` = Pneumonia
- Preprocessing:
  - Resized to 299×299 for InceptionV3
  - Normalized and converted to 3-channel tensors

---

## 🧠 Model: InceptionV3 (Transfer Learning)
- Pretrained on ImageNet
- Final FC layer modified to output 2 logits (Normal/Pneumonia)
- All base layers frozen
- Only classification head trained

---

## ⚙️ Training Setup
- Loss Function: Weighted CrossEntropyLoss (to handle class imbalance)
- Optimizer: Adam (`lr=1e-4`)
- Epochs: 2 (for demonstration)
- Hardware: T4 GPU (Google Colab)

---

## 📈 Evaluation Metrics
| Metric     | Score   |
|------------|---------|
| Accuracy   | 75.32%  |
| Precision  | 72.26%  |
| Recall     | 98.21%  |

### Confusion Matrix:
```
[[87, 147],
 [ 7, 383]]
```

---

## 🔍 Class Imbalance Handling
- Detected by counting training labels using `Counter`
- Found pneumonia class dominated dataset
- Used inverse-frequency-based class weights in CrossEntropyLoss

---

## 🧪 Overfitting Prevention
- Used pretrained weights with frozen feature extractor
- Normalized input images
- Data shuffled in batches
- Optional: Dropout and Augmentation could be added in later phases

---

## 📦 Files Included
- `inceptionv3_pneumonia.pth`: Saved model weights
- `ICMR_Pneumonia_Presentation.pptx`: 3-slide summary presentation
- `ICMR_Pneumonia_Steps1to3.ipynb`: Training pipeline (Steps 1–3)
- `README.md`: Project description and evaluation summary

---

## 🗣️ Author
**Sachin Dahiya** – ICMR Project Research Scientist-I Assessment Task
