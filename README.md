# 😷 Face Mask Detection using CNN & OpenCV

## 📌 Project Overview
This project implements a **Face Mask Detection System** using **Convolutional Neural Networks (CNN)** and **Computer Vision**.  
The model classifies whether a person is **wearing a face mask or not** and supports **real-time detection using a webcam**.

The project demonstrates a complete **end-to-end Machine Learning workflow**, including:
- Dataset handling
- Model training
- Model evaluation
- Real-time deployment

---

## 🎯 Problem Statement
Manual monitoring of face mask compliance in public places is inefficient and not scalable.  
This project provides an **automated deep learning–based solution** for detecting face mask usage.

---

## 🧠 Model Performance
- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~94%

### Confusion Matrix Summary
- Correct predictions (With Mask): 3657
- Correct predictions (Without Mask): 3762
- Very low misclassification rate

---

## 📊 Results
Model evaluation outputs are stored in the `results/` folder:
- Confusion Matrix
- Accuracy vs Epochs graph
- Loss vs Epochs graph

---

---

## ⚙️ Technologies Used
- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib

---

## 📁 Dataset Information
- Source: Kaggle Face Mask Dataset
- Classes:
  - `with_mask`
  - `without_mask`
- Image size: 128 × 128
- Train–test split: 80% / 20%

> Only **sample images** are uploaded to GitHub for reference.

---

## 📥 Model & Training History (IMPORTANT)

Due to GitHub file size limitations, the trained model and training history files are hosted on **Google Drive**.

### 🔗 Download Links
- **Trained Model (`mask_detector_model.h5`)**  
  👉 [PASTE_YOUR_GOOGLE_DRIVE_MODEL_LINK_HERE](https://drive.google.com/file/d/1T6NeLv9Q18rK7p2LdmSaAdrHF6rHvAG6/view?usp=sharing)

### Running Commands

 Step 1: Clone the repository
git clone https://github.com/Tejaswini-ece/Face_Mask_Detection.git
cd Face_Mask_Detection

Step 2: Create virtual environment
python -m venv venv

Step 3: Activate virtual environment
# For Windows
venv\Scripts\activate

For Linux / macOS
source venv/bin/activate

Step 4: Install required dependencies
pip install -r requirements.txt

Step 5: (Optional) Train the model
python train.py

Step 6: Evaluate the model
python evaluation.py

Step 7: Run real-time face mask detection
python detect.py

Step 8: Deactivate virtual environment (optional)
deactivate

### ⚠️ Important Notes

Virtual environment (venv/) is not uploaded to GitHub

Model and pickle files must be downloaded manually

Webcam access is required for real-time detection

Lighting and face angle may affect accuracy

### 🌍 Applications

Public safety monitoring

Smart surveillance systems

Workplace safety enforcement

Educational machine learning projects

### 📚 Learning Outcomes

Through this project, I learned:

CNN-based image classification

Image preprocessing techniques

Model evaluation using confusion matrix

Real-time deployment with OpenCV

Proper GitHub project structuring for ML projects

👤 Author

Theeparthi Tejaswini
Electronics & Communication Engineering Student
Interests: Machine Learning, Embedded Systems


