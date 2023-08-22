import os
import numpy as np

from werkzeug.utils import secure_filename
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

app = Flask(__name__)
MODEL_PATH = 'model.h5'

model = load_model(MODEL_PATH)


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)

    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    if preds.max() < 0.3 or np.count_nonzero(preds == 0) < 5:
        preds = {
            "name": "Error",
            "properties":"Image uploaded is not of a leaf",
        }
    else:
        preds = np.argmax(preds, axis=1)
        if preds == 0:
            preds = {
                "name": "Asthma Plant",
                "properties": "Euphorbia is used for breathing disorders including asthma, bronchitis, and chest congestion.",
            }
        elif preds == 1:
            preds = {
                "name": "Avaram",
                "properties": "Prevents bacterial growth and is also effective in curing infections",
            }
        elif preds == 2:
            preds = {
                "name": "coatbuttons",
                "properties": " Useful in jaundice, bronchial catarrh, diarrhoea, dysentery, inflammation, ulcers, anal fistula, and hemorrhoids",
            }
        elif preds == 3:
            preds = {
                "name": "heart-leaved moonseed",
                "properties": "The plant is of great interest to researchers across the globe because of its reported medicinal properties like anti-diabetic, anti-periodic, anti-spasmodic, anti-inflammatory, anti-arthritic, anti-oxidant, anti-allergic, anti-stress",
            }
        elif preds == 4:
            preds = {
                "name": "Indian Jujube",
                "properties": "A powerhouse of antioxidants jujube fruits are well known to enhance skin health, detoxifies the blood and improves cardiac function",
            }
        elif preds == 5:
            preds = {
                "name": "Malabar Catmint",
                "properties": "Folkloric medicine to treat amentia, anorexia, fevers, swellings, rheumatism ",
            }
        elif preds == 6:
            preds = {
                "name": "Mexican Mint",
                "properties": "Improve the health of your skin, detoxify the body, defend against colds, ease the pain of arthritis, relieve stress and anxiety, treat certain kinds of cancer, and optimize digestion",
            }
        elif preds == 7:
            preds = {
                "name": "Panicled Foldwing",
                "properties": "Leaves bestowed with essential minerals and vitamins helps to soothe the upset gut, treats asthma, common cold and sore throat",
            }
        elif preds == 8:
            preds = {
                "name": "Prickly Chaff Flower",
                "properties":   "Achyranthes Aspera is used in the treatment of boils, asthma, in facilitating delivery, bleeding, bronchitis, debility, dropsy, cold.",
            }
        elif preds == 9:
            preds = {
                "name": "Punarnava",
                "properties":   "The herb can be used as a diuretic in kidney disorders and helps to manage symptoms of spleen enlargement.",
            }
        elif preds == 10:
            preds = {
                "name": "Rosary Pea",
                "properties": "The seeds of the rosary pea have been used to make beaded jewelry, which can lead to abrin poisoning if the seeds are swallowed. Abrin has some potential medical uses, such as in treatment to kill cancer cells.",
            }
        elif preds == 11:
            preds = {
                "name": "Sweet flag",
                "properties": "Weet flag is mainly used in medicine. The oil is used to cure gastritis. In the form of infusion it is carminative and possesses emetic and anti-spasmodic properties. It is used in perfumery industry.",
            }
        elif preds == 12:
            preds = {
                "name": "Tinnevelly Senna",
                "properties":   "The leaves and the fruit of the plant are used to make medicine. Senna is an FDA-approved nonprescription laxative.",
            }
        elif preds == 13:
            preds = {
                "name": "Trellis Vine",
                "properties": "The trellis-vine powder reduces blood sugar levels, treats jaundice and other minor wounds, and has many medicinal uses.",
            }
        elif preds == 14:
            preds = {
                "name": "Velvet bean",
                "properties": "The plant's leaf extracts and seeds show promise for the treatment of Parkinson's disease, male infertility, and nervous disorders, mainly due to their antioxidant (free radical scavenging) effects.",
            }
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'static', 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)
        result = preds
        return render_template('result.html', result=result, image_path="/static/uploads/" + secure_filename(f.filename))
    return None


if __name__ == '__main__':
    app.run(debug=True)
