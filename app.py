import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename          ## for file path (remove / from path)


app = Flask(__name__)

model =tf.keras.models.load_model('model.h5',compile=False)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(64, 64))  ##  image load  image is 64 * 64
    x = image.img_to_array(img)     ## convert into array [neural network accept only array]
    x = np.expand_dims(x, axis=0)   ## to skip dimension error                 axis 0 to be image as row
    x = np.array(x, 'float32')      
    x /= 255                       ## to return number prediction from 1 to -1
    preds = model.predict(x)      ## prediction to this image
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction567
        preds = model_predict(file_path, model)
        print(f'#########################{preds}################')

        disease_class = ['You Have Covid-19','Non Covid-19']
        a = preds[0]
        ind=np.argmax(a)
        print(f"###############{ind}###############")
        print('Prediction:', disease_class[ind])
        result=disease_class[ind]
        def numR () :
            if a[0] > a[1] :
                return round(a[0] * 100 , 1)
            else :
                return round(a[1] * 100 , 1)
        
            
        return f"{result} And Accuracy Is %{numR()}"
    return None


if __name__ == '__main__':
    app.run(debug=True)
