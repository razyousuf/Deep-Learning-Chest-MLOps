from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS, cross_origin
from cnnClassifier.pipeline.predict import PredictionPipeline
from cnnClassifier.utils.common import decodeImage


# 1. Initialize the Flask application
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


# 2. Create a client Class
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


# 3. Create a default route
@app.route('/', methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')

# 4. Create a route for training the model
@app.route('/train', methods=['POST'])
@cross_origin()
def trainRoute():
    os.system("python main.py")
    os.system("dvc repro")
    return "Model Trained Successfully"


# 5. Create a route for prediction
@app.route('/predict', methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clientApp.filename)
    result = clientApp.classifier.predict()
    return jsonify(result)


# Run the application
if __name__ == '__main__':
    clientApp = ClientApp()
    app.run(host='0.0.0.0', port=8080, debug=True)