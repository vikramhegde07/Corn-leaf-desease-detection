import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from skimage import transform


def get_model():
    cnn = load_model('static/Model EfficientNet.keras')
    return cnn


def predict(image_data):
    loaded_model = get_model()
    img = img_to_array(image_data)
    np_image = transform.resize(img, (224, 224, 3))
    image4 = np.expand_dims(np_image, axis=0)
    result__ = loaded_model.predict(image4)
    return result__
