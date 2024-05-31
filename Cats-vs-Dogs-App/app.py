from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = load_model('cats_dogs_classifier_model.h5')


def load_and_prepare_image(img_path):
    """Load and prepare an image file to an appropriate format for model prediction."""
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.0


@app.route("/", methods=["GET"])
def home():
    """Render the main page with the upload form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle the image upload and perform prediction using the pre-trained model."""
    if request.method == "POST":
        # Check if the post request has the file part
        if 'image' not in request.files:
            return render_template("index.html", prediction="No file part")

        file = request.files['image']

        if file.filename == '':
            return render_template("index.html", prediction="No selected file")

        if file and allowed_file(file.filename):
            # Save the file to a temporary file
            filepath = os.path.join('static/images', file.filename)
            file.save(filepath)

            # Prepare image for prediction
            prepared_image = load_and_prepare_image(filepath)

            # Make a prediction
            predictions = model.predict(prepared_image)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index] * 100

            # Get the class labels from the train_generator (used during training)
            class_labels = ['Cat','Dog']  # Update this with actual class names

            # Return the result to the user
            return render_template("index.html",
                                   prediction=class_labels[predicted_class_index],
                                   confidence=f"{confidence:.2f}%")


def allowed_file(filename):
    """Check if the uploaded file is allowed."""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif']


if __name__ == "__main__":
    app.run(debug=True)
