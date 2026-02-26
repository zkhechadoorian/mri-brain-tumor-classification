# 🧠 MRI Brain Tumor Classification

This project demonstrates a complete deep learning pipeline for classifying brain MRI scans into different tumor categories (glioma, meningioma, pituitary, or no tumor).  
The project emphasizes **medical data understanding**, **model interpretability**, and **ethical AI use**.


## 🏗️ Model Architecture
This project uses the ResNet18 convolutional neural network architecture. ResNet18 consists of:
- An initial convolutional layer 
- Four sequential blocks, each containing two "BasicBlocks" (each block has two convolutional layers and a shortcut connection) (16 layers)
- A global average pooling layer
- A fully connected output layer

There are a total of 18 layers. 
The shortcut (residual) connections help prevent vanishing gradients, enabling deeper networks to learn effectively. The model is adapted for grayscale MRI images and trained with class weights to address imbalance.

## ⚖️ Why Class Imbalance Matters
Class imbalance occurs when some classes (e.g., 'no_tumor') are underrepresented compared to others. In medical imaging, this can cause the model to be biased toward predicting the majority classes, potentially missing critical cases. To mitigate this, class weights are used in the loss function, penalizing errors on minority classes more heavily and improving diagnostic reliability.


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
│   ├── Training/
│   ├── Testing/
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

All data is anonymized and used for educational purposes only. An image of the class distribution is shown below after the data is cleaned. Most notably, the 'no_tumor' class contains fewer images than any of the tumor-containing classes. As a result, class weights are implemented into the loss calculation to penalize misclassifications of 'no_tumor' images more heavily during training. 

![class distribution](/assets/class_distribution.png)



---

## 🧠 Methodology
1. **Data Exploration**
    - [01_explore_data.ipynb](notebooks/01_explore_data.ipynb): Visualizes dataset structure, class distribution, image quality, and outlier detection.
2. **Preprocessing**
    - [02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb): Resizes images, applies CLAHE for contrast normalization, removes duplicates, handles class imbalance, and creates train/val/test splits.
3. **Model Training and Evaluation**
    - [03_train_model.ipynb](notebooks/03_train_model.ipynb): Implements and trains ResNet18 with class weights, tracks training/validation curves, and saves model weights. It also computes precision, recall, F1, ROC, confusion matrix, and analyzes false positives/negatives.
4. **Explainability**
    - [05_explainability.ipynb](notebooks/05_explainability.ipynb): Applies Grad-CAM to visualize model attention and interpret predictions.
5. **Ethics & Limitations discussion**
    - Discusses dataset bias, clinical limitations, and ethical considerations for AI in medicine.

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
