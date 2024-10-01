import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing.image import img_to_array
from skimage import transform


def get_model(model_no):
    if model_no == 1:
        model = load_model('static/Models/Model EfficientNet.keras')
    elif model_no == 2:
        model = load_model('static/Models/Model CNN.keras')
    else:
        model = load_model('static/Models/Model VGG19.keras')
    return model


def predict(image_data,model_no):
    loaded_model = get_model(model_no)
    img = img_to_array(image_data)
    np_image = transform.resize(img, (224, 224, 3))
    image4 = np.expand_dims(np_image, axis=0)
    result__ = loaded_model.predict(image4)
    return result__
