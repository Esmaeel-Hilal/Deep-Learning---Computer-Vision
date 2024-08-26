from flask import Flask, render_template, request, url_for
import os
import tensorflow as tf
from keras.models import load_model


class_names_file_path = "/home/esmaeel-hi/colab_env_folder/07_milestone_project_1_food_vision/flask_deployment_project/class_names.txt"
model_path = "/home/esmaeel-hi/colab_env_folder/07_milestone_project_1_food_vision/fine_tune_saved_model_b0"
upload_folder = "static/uploads/"


# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder


# Load the model and class names
model = load_model(model_path)


def get_class_names():
  with open(class_names_file_path, 'r') as f:
        return [line.strip() for line in f]


class_names = get_class_names()


# Image preparation function
def load_and_prepare_image(file_name, img_shape=224, scale=True):
  img = tf.io.read_file(file_name)
  img = tf.io.decode_image(img, channels=3)
  img = tf.image.resize(img, [img_shape, img_shape])
  return img / 255.0 if scale else img


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    
    # Ensure the uploads folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Save the image in the static/uploads directory
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
    imagefile.save(image_path)

    # Prepare image for prediction
    img = load_and_prepare_image(image_path, scale=False)
    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    y_pred = pred_prob.argmax()
    pred_conf = pred_prob[0][y_pred]
    pred_class_name = class_names[y_pred]

    classification = f'{pred_class_name} ({pred_conf * 100:.2f}%)'

    # Generate URL for the image to display it after prediction
    image_url = url_for('static', filename=f'uploads/{imagefile.filename}')

    return render_template('index.html', prediction=classification, image_url=image_url)


if __name__ == '__main__':
    app.run(port=3000, debug=True)