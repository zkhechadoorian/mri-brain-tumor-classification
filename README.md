# 🧠 MRI Brain Tumor Classification

This project demonstrates a complete deep learning pipeline for classifying brain MRI scans into different tumor categories (glioma, meningioma, pituitary, or no tumor).  
The project emphasizes **medical data understanding**, **model interpretability**, and **ethical AI use**.

---

## 🚀 Objectives
- Explore and preprocess MRI brain images.
- Train a convolutional neural network (CNN) using transfer learning.
- Evaluate performance with medical metrics (sensitivity, specificity, ROC-AUC).
- Apply explainability techniques (Grad-CAM) to interpret model predictions.
- Discuss limitations, bias, and ethical considerations.

---

## Project Structure
```
mri-brain-tumor-classification/
│
├── README.md
├── requirements.txt
│
├── data/
│   └── (add your MRI dataset here — not tracked by Git)
│
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_model.ipynb
│   ├── 04_evaluate_model.ipynb
│   └── 05_explainability.ipynb
│
├── models/
│   └── (saved trained models go here)
│
├── utils/
│   ├── dataloader.py
│   └── preprocessing.py
│
└── results/
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── gradcam_example.png
```

## 🩺 Dataset
**Source:** [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)  
**Description:**  
Contains MRI images classified into four groups:
- Glioma Tumor  
- Meningioma Tumor  
- Pituitary Tumor  
- No Tumor  

All data is anonymized and used for educational purposes only.

---

## 🧠 Methodology
1. Data Exploration (visualize and understand MRI scans)  
2. Preprocessing (resize, normalize, augment)  
3. Model Training (EfficientNet/ResNet transfer learning)  
4. Evaluation (precision, recall, ROC, confusion matrix)  
5. Explainability (Grad-CAM visualization)  
6. Ethics & Limitations discussion  

---

## 🧪 Results (example placeholder)
| Metric | Value |
|--------|--------|
| Accuracy | 94.2% |
| Sensitivity | 91.8% |
| Specificity | 95.5% |

Example Grad-CAM heatmap:  
![Grad-CAM Example](results/gradcam_example.png)

---

## ⚖️ Ethical Statement
This model is **for research and educational purposes only**.  
It is **not** a diagnostic or clinical decision tool.  
All data used is publicly available and de-identified.

---

## 📚 References
- Bhuvaji et al., *Brain Tumor Classification (Kaggle)*  
- Rajpurkar et al., *AI in Radiology: The Challenges of Generalization* (Nature Medicine, 2022)
