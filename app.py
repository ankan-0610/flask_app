# python_server.py
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import io
import base64
import cv2
import requests
import numpy as np
import os, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import load

# Create Flask app

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/generate_plot', methods=['POST'])
def generate_plot():
    # Get image URLs and user inputs from the request
    data = request.get_json()
    image_urls = data['imageUrls']
    user_inputs = data['userInputs']

    GB_ratio = []
    RB_ratio = []
    RG_ratio = []
    BG_ratio = []
    BR_ratio = []
    GR_ratio = []
    Conc = np.array(user_inputs, dtype=float)

    # loop through the urls
    for image_url in image_urls:
        # Get the image
        # url_path = urlparse(image_url).path
        # filename = os.path.basename(url_path)
        # Read the image using OpenCV
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            GB_ratio.append(get_GB(img))
            RB_ratio.append(get_RB(img))
            RG_ratio.append(get_RG(img))
            BG_ratio.append(get_BG(img))
            BR_ratio.append(get_BR(img))
            GR_ratio.append(get_GR(img))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image: {e}")
            return
    GB_ratio = np.array(GB_ratio)
    RB_ratio = np.array(RB_ratio)
    RG_ratio = np.array(RG_ratio)
    BG_ratio = np.array(BG_ratio)
    BR_ratio = np.array(BR_ratio)
    GR_ratio = np.array(GR_ratio)

    GB_ratio, RB_ratio, RG_ratio, BG_ratio, BR_ratio, GR_ratio, Conc = sort_arrays(GB_ratio, 
                                    RB_ratio, RG_ratio, BG_ratio, BR_ratio, GR_ratio, Conc)

    RB_Model = RandomForestRegressor(n_estimators=100, random_state=42)
    RB_Model.fit(RB_ratio.reshape(-1, 1), Conc)
    GB_Model = RandomForestRegressor(n_estimators=100, random_state=42)
    GB_Model.fit(GB_ratio.reshape(-1, 1), Conc)
    RG_Model = RandomForestRegressor(n_estimators=100, random_state=42)
    RG_Model.fit(RG_ratio.reshape(-1, 1), Conc)

    # Calculate R-squared scores
    r2_RB = r2_score(Conc, RB_Model.predict(RB_ratio.reshape(-1, 1)))
    r2_GB = r2_score(Conc, GB_Model.predict(GB_ratio.reshape(-1, 1)))
    r2_RG = r2_score(Conc, RG_Model.predict(RG_ratio.reshape(-1, 1)))

    # Choose the model with the highest R-squared score
    x_values = None
    y_label = None
    if r2_RB >= r2_GB and r2_RB >= r2_RG:
        best_model = RB_Model
        x_values = RB_ratio
        y_label = 'R/B ratio'
        ratio_num=0
    elif r2_GB >= r2_RB and r2_GB >= r2_RG:
        best_model = GB_Model
        x_values = GB_ratio
        y_label = 'G/B ratio'
        ratio_num=1
    else:
        best_model = RG_Model
        x_values = RG_ratio
        y_label = 'R/G ratio'
        ratio_num=2

    predictions = best_model.predict(x_values.reshape(-1, 1))

    plt.scatter(Conc, x_values)
    plt.plot(predictions, x_values)
    plt.xlabel("Conc in uM")
    plt.ylabel(y_label)

    # Your logic to generate plot using Matplotlib
    # Example: (Replace this with your actual logic)
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    # plt.xlabel('x')
    # plt.ylabel('y')

    # Save plot image to bytes
    image_bytes = io.BytesIO()
    plt.savefig(image_bytes, format='png')
    image_bytes.seek(0)

    # Convert image to base64 for sending in JSON response
    image_base64 = base64.b64encode(image_bytes.read()).decode('utf-8')

    plt.close()

    # Example table data (Replace this with your actual data)
    # Create table_data by dynamically copying values from numpy arrays
    table_data = [{'Actual Concentration': str(value1), 
                    'Predicted Concentration': str(value2)} 
                    for value1, value2 in zip(Conc, predictions)]

    return jsonify({'plotImage': image_base64, 'tableData': table_data})


def get_GB(img):
    b, g, r = cv2.split(img)
    return np.mean(g) / np.mean(b)

def get_RB(img):
    b, g, r = cv2.split(img)
    return np.mean(r) / np.mean(b)

def get_RG(img):
    b, g, r = cv2.split(img)
    return np.mean(r) / np.mean(g)

def get_BG(img):
    b, g, r = cv2.split(img)
    return np.mean(b) / np.mean(g)

def get_BR(img):
    b, g, r = cv2.split(img)
    return np.mean(b) / np.mean(r)

def get_GR(img):
    b, g, r = cv2.split(img)
    return np.mean(g) / np.mean(r)

def sort_arrays(GB_ratio, RB_ratio, RG_ratio, BG_ratio, BR_ratio, GR_ratio, Conc):
    # Create pairs of elements from both arrays
    pairs = list(zip(Conc, GB_ratio, RB_ratio, RG_ratio, BG_ratio, BR_ratio, GR_ratio))

    # Sort the pairs based on the first element of each pair
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    Conc_np = np.array([])
    RB_ratio = np.array([])
    GB_ratio = np.array([])
    RG_ratio = np.array([])
    BG_ratio = np.array([])
    BR_ratio = np.array([])
    GR_ratio = np.array([])
    for pair in sorted_pairs:
        Conc_np = np.append(Conc_np,pair[0])
        GB_ratio = np.append(GB_ratio, pair[1])
        RB_ratio = np.append(RB_ratio, pair[2])
        RG_ratio = np.append(RG_ratio, pair[3])
        BG_ratio = np.append(BG_ratio, pair[4])
        BR_ratio = np.append(BR_ratio, pair[5])
        GR_ratio = np.append(GR_ratio, pair[6])
    return GB_ratio, RB_ratio, RG_ratio, BG_ratio, BR_ratio, GR_ratio, Conc_np

# if __name__ == '__main__':
#     app.run()

# # Load the model from the pickle file
# with open('your_model.pkl', 'rb') as f:
#     best_model = pickle.load(f)

import firebase_admin
from firebase_admin import credentials, storage

# Load the service account JSON from environment variable
service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
cred = credentials.Certificate(service_account_info)
firebase_admin.initialize_app(cred, {
    'storageBucket': 'image-poc-1ba68.appspot.com'
})

# Download model from Firebase Storage
def download_model(model_name):
    bucket = storage.bucket()
    blob = bucket.blob(f'ml_models/{model_name}')
    model_path = f'{model_name}'
    blob.download_to_filename(model_path)
    return model_path

def get_model(model_name):
    local_model_path = f'{model_name}'
    if not os.path.exists(local_model_path):
        return download_model(model_name)
    return local_model_path

# Load the joblib model after ensuring it's downloaded
def load_model(model_name):
    model_path = get_model(model_name)  # Ensure the model is downloaded
    print(f"Loading model from {model_path}...")
    
    model = load(model_path)
    
    return model

@app.route( '/predict_result', methods=['POST'] )
def predict_result():
    data = request.get_json()
    image_urls = data['imageUrls']
    prediction_results = []

    model = load_model('BR_model.joblib')

    # loop through the urls
    for idx, image_url in enumerate(image_urls):
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Raise an HTTPError for bad responses
            image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            gray_image = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)

            b, g, r = cv2.split(img)
            # rgb_values = [np.mean(r), np.mean(g), np.mean(b)]
            g0 = 154.45386363636365

            feature = g0/np.mean(g)

            conc = model.predict(feature)

            prediction_results.append({
                'image_url': image_url,
                'conc': round(conc, 2),
                'grayscale': int(round(np.mean(gray_image))),
            })

        except requests.exceptions.RequestException as e:
            print(f"Error fetching image {idx}: {e}")
            return

    return jsonify(prediction_results)