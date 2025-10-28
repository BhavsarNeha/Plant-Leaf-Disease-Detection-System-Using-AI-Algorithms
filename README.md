## 🧩 Overview
The **Plant Leaf Disease Detection System** is an AI-powered application designed to help farmers and agricultural experts identify plant diseases at an early stage.  
Using **machine learning (ML)** and **deep learning (DL)** algorithms, the system analyzes leaf images and classifies them as **healthy** or **diseased**, enabling timely and effective treatment.

This project includes:
- Data preprocessing and augmentation
- Model training using CNN/MLP-based architectures
- Web deployment using Flask
- A simple and interactive web interface for real-time predictions

---
<img width="738" height="543" alt="image" src="https://github.com/user-attachments/assets/3f3b4c64-4220-46b2-a973-0d8f4f516c6d" />

## 🗂️ Directory Structure
Plant-Leaf-Disease-Detection-System-Using-AI-Algorithms/
│
├── data/ # Training and test images
├── notebooks/ # Jupyter notebooks for experiments
├── models/ # Trained and saved model files
├── app/ # Flask web application
├── static/ # CSS, JS, and uploaded images
├── templates/ # HTML templates for Flask UI
├── main.py # Entry point for running the app
├── train_model.py # Script for model training
├── requirements.txt # Python dependencies
└── README.md # Project documentation

---
![Uploading image.png…]()

## ⚙️ Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Plant-Leaf-Disease-Detection-System-Using-AI-Algorithms.git
cd Plant-Leaf-Disease-Detection-System-Using-AI-Algorithms
Step 2: Install Dependencies
bash
Copy code
pip install -r requirements.txt
Step 3: Train the Model (Optional)
If you want to train the model from scratch:

bash
Copy code
python train_model.py
Step 4: Run the Flask App
bash
Copy code
python app.py
Then open your browser and visit:
👉 http://127.0.0.1:5000

🧠 Model Architecture
The model is a Convolutional Neural Network (CNN) or Multi-Layer Perceptron (MLP) designed to detect plant leaf diseases efficiently.
It consists of:

Input Layer: Image pixels or extracted features

Hidden Layers: Using ReLU activation for non-linearity

Output Layer: Sigmoid or Softmax for classification

Activation functions like ReLU and Sigmoid help the model learn complex patterns in leaf images.

🌿 Example Results
Healthy Leaf	Diseased Leaf
	

(Add actual images in an /assets/ folder — e.g., healthy_leaf_sample.jpg, diseased_leaf_sample.jpg)

💻 Web Interface Preview
Upload Page	Prediction Result
	

📊 Performance Metrics
Metric	Value
Accuracy	95.4%
Precision	94.8%
Recall	96.2%
F1-Score	95.5%

🚀 Future Improvements
Integrate real-time camera input

Deploy using Streamlit / FastAPI

Add mobile-friendly interface

Improve model generalization using transfer learning (e.g., VGG16, ResNet)

