from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.predict import PredictionPipeline
from cnnClassifier.utils.common import decodeImage
import json
from yaml import dump
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as dq
from ruamel.yaml import YAML

# Set environment for Unicode handling
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Prediction pipeline object
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clientApp = ClientApp()

# ---------------------- ROUTES ------------------------

@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return render_template("base.html")  # Main landing or intro page

@app.route("/predict", methods=["GET"])
@cross_origin()
def home():
    return render_template("home.html")  # Image upload/predict interface

@app.route("/train-ui", methods=["GET"])
@cross_origin()
def train_ui():
    return render_template("train.html")  # UI to send training parameters


@app.route("/about", methods=["GET"])
@cross_origin()
def about():
    return render_template("about.html")


@app.route('/train', methods=['GET', 'POST'])
@cross_origin()
def trainRoute():
    scores = None
    if os.path.exists("scores.json"):
        with open("scores.json", "r") as f:
            scores = json.load(f)

    if request.method == 'POST':
        try:
            # 1. Parse form values
            config = {
                "IS_AUGMENTATION": request.form.get("IS_AUGMENTATION") == "True",
                "BATCH_SIZE": int(request.form.get("BATCH_SIZE")),
                "EPOCHS": int(request.form.get("EPOCHS")),
                "LEARNING_RATE": float(request.form.get("LEARNING_RATE")),
                "IMAGE_SIZE": [224, 224, 3],
                "INCLUDE_TOP": False,
                "WEIGHTS": dq("imagenet"),
                "CLASSES": 2 
        }
            yaml = YAML()
            yaml.default_flow_style = True
            # 2. Write to params.yaml
            with open("params.yaml", "w") as f:
                yaml.dump(config, f)#, default_flow_style=True)

            # 3. Trigger training
            os.system("python main.py")
           # os.system("dvc repro")

            return render_template('train.html', status='success', scores=scores)

        except Exception as e:
            print("Training error:", str(e))
            return render_template('train.html', status='error')

    # GET request: just show form
    return render_template('train.html', scores=scores)


@app.route("/predict", methods=["POST"])
@cross_origin()
def predictRoute():
    try:
        image = request.json["image"]
        decodeImage(image, clientApp.filename)
        result = clientApp.classifier.predict()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
