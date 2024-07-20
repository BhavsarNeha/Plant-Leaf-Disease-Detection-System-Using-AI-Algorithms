# Plant Leaf Disease Detection System

## Overview
This project aims to develop a Plant Leaf Disease Detection System using AI algorithms to detect diseases in plant leaves. The system includes data collection, model training, and a user-friendly web interface for farmers.

## Directory Structure
- `data/`: Contains training and test images.
- `notebooks/`: Jupyter notebooks for experiments.
- `models/`: Trained models.
- `app/`: Flask application.
- `main.py`: Main script to run the application.
- `requirements.txt`: Python dependencies.

## Installation
1. Clone the repository:
    ```
    git clone https://github.com/yourusername/Plant_Leaf_Disease_Detection.git
    ```
2. Install the dependencies:
    ```
    pip install -r requirements.txt
    ```

## Usage
1. Prepare the dataset and place it in the `data/train` directory.
2. Run the model training script:
    ```
    python train_model.py
    ```
3. Start the Flask application:
    ```
    python app.py
    ```
4. Open your browser and go to `http://127.0.0.1:5000` to use the application.

## License
This project is licensed under the MIT License.
