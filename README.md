# 🫁 Pulmonary Nodule Detection using Deep Learning

This project focuses on **early detection of pulmonary nodules** (possible signs of lung cancer) from CT scan images using **deep learning techniques**. The system classifies CT images into categories such as **benign, malignant, or unlabeled**, helping in assisting radiologists for faster and more accurate diagnosis.

---

## 📌 Features
- Preprocessed **lung CT scan dataset**.
- Deep learning model for **pulmonary nodule detection**.
- Trained using **TensorFlow/Keras + CNN** (Convolutional Neural Networks).
- **Streamlit web app** for interactive image upload and detection.
- **Grad-CAM explainability** to visualize affected regions in lung scans.
- Scalable and extendable for real-world deployment.

---

## 📂 Project Structure

Pulmonary_Nodule_Detection/
│
 # Source code files
│ ── train.py # Training script
│ ── predict.py # Inference script
│ 
│
| ─ models/ # Saved trained models (not uploaded to GitHub)
│
├── sample_images/ # Few small sample images for testing
│
├── Lung_Cancer_Prediction.v2i.folder/ # Full dataset (Not in GitHub, download separately)
│
├── app.py # Streamlit web app
| 
|
└── .gitignore # Ignored files (datasets, large models, cache)



---

## 📊 Dataset

The dataset used for this project consists of **CT scan images** categorized into:  
- `Benign`  
- `Malignant`  
- `Unlabeled`

🔗 **Download Dataset Here:**  
👉  https://drive.google.com/drive/folders/13rm43Xh3vW7hnpfQD-7whYM9_kUgzCvY?usp=drive_link

> ⚠️ Note: The dataset is **not stored in this repository** due to large file size. Only small sample images are included.

---

## ⚙️ Installation & Setup

1. **Clone this repository**
   ```bash
   git clone https://github.com/HarshaVardhan4223/Pulmonary_nodule_detection.git
   cd Pulmonary_nodule_detection

   Create virtual environment:

   python -m venv venv
source venv/Scripts/activate   # On Windows
source venv/bin/activate       # On Linux/Mac

Download dataset from the Google Drive link
 and place it inside  --   Pulmonary_Nodule_Detection/Lung_Cancer_Prediction.v2i.folder/

🚀 Usage
1️⃣ Training the Model:

python train.py

2️⃣ Running Predictions:

python predict.py --image_path sample_images/test_image.jpg

3️⃣ Running the Streamlit Web App:

streamlit run app.py


This will launch a web-based interface where you can upload lung CT scan images and get predictions along with Grad-CAM visualizations.


📈 Model Performance

Accuracy: ~XX% (update with your results)

Precision: XX%

Recall: XX%

F1-score: XX%


🧠 Explainability with Grad-CAM

The project integrates Grad-CAM (Gradient-weighted Class Activation Mapping) to highlight the regions in CT scans that influence the model’s predictions, improving trust and interpretability.

🔮 Future Work

Improve accuracy with 3D CNNs for volumetric CT data.

Deploy using FastAPI/Flask as a backend for mobile apps.

Extend to multi-class classification (different lung diseases).

Integrate with Flutter/React Native mobile app.


🙌 Contributors

Singarapu Harshavardhan – Project Development & Deployment


📜 License

This project is licensed under the MIT License – feel free to use and modify with proper attribution.


⭐ Acknowledgements

Dataset source: Roboflow

TensorFlow/Keras for deep learning

Streamlit for web app



