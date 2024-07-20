from flask import Flask, request, render_template
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/leaf_disease_model.h5')

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        file_path = f"app/static/{file.filename}"
        file.save(file_path)
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return render_template('result.html', predicted_class=predicted_class)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
