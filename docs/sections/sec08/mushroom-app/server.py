import os

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, Query, File
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

AUTOTUNE = tf.data.experimental.AUTOTUNE

prefix = ""

# Setup FastAPI app
app = FastAPI(
    title="API Server",
    description="API Server",
    version="v1",
    openapi_url="/api/openapi.json"
)

# Enable CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a model from disk
best_model = "models/mobilenetv2_train_baseTrue_1618955507.hdf5"
prediction_model = tf.keras.models.load_model(best_model)
label_names = ['oyster', 'crimini', 'amanita']
# Create label index for easy lookup
label2index = dict((name, index) for index, name in enumerate(label_names))
index2label = dict((index, name) for index, name in enumerate(label_names))

# Mount frontend app
app.mount("/app", StaticFiles(directory="app"), name="app")


# Routes
@app.get(
    "/api",
    summary="Index",
    description="Root api"
)
async def get_index():
    return {
        "message": "Welcome to the API Service"
    }


# Additional routers here
@app.get(
    "/api/test_prediction",
    summary="Test Prediction",
    description="Test Prediction"
)
async def get_test_prediction(
        image: str = Query(..., description="Image to use for prediction")
):
    print("image", image)

    image_path = os.path.join("app", "images", image)

    # Load & preprocess
    test_data = await load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = prediction_model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = index2label[idx]

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction) * 100, 2)
    }


@app.post(
    "/api/predict_file",
    summary="Test Prediction",
    description="Test Prediction"
)
async def predict_file(
        file: bytes = File(...)
):
    print("predict_file")
    print(len(file), type(file))

    # Save the image
    image_path = "test.png"
    with open(image_path, "wb") as output:
        output.write(file)

    # Load & preprocess
    test_data = await load_preprocess_image_from_path(image_path)

    # Make prediction
    prediction = prediction_model.predict(test_data)
    idx = prediction.argmax(axis=1)[0]
    prediction_label = index2label[idx]

    return {
        "input_image_shape": str(test_data.element_spec.shape),
        "prediction_shape": prediction.shape,
        "prediction_label": prediction_label,
        "prediction": prediction.tolist(),
        "accuracy": round(np.max(prediction) * 100, 2)
    }


# Util functions
async def load_preprocess_image_from_path(image_path):
    print("Image", image_path)

    image_width = 224
    image_height = 224
    num_channels = 3

    # Prepare the data
    def load_image(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=num_channels)
        image = tf.image.resize(image, [image_height, image_width])
        return image

    # Normalize pixels
    def normalize(image):
        image = image / 255
        return image

    test_data = tf.data.Dataset.from_tensor_slices(([image_path]))
    test_data = test_data.map(load_image, num_parallel_calls=AUTOTUNE)
    test_data = test_data.map(normalize, num_parallel_calls=AUTOTUNE)
    test_data = test_data.repeat(1).batch(1)

    return test_data
