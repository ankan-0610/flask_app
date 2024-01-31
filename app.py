# python_server.py
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import io
import base64
import cv2
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

    RB_ratio, Conc = get_analysis(RB_ratio, Conc)

    RB_Model = LinearRegression()
    RB_Model.fit(RB_ratio.reshape(-1, 1), Conc)
    GB_Model = LinearRegression()
    GB_Model.fit(GB_ratio.reshape(-1, 1), Conc)
    RG_Model = LinearRegression()
    RG_Model.fit(RG_ratio.reshape(-1, 1), Conc)

    # Calculate R-squared scores
    r2_RB = r2_score(Conc, RB_Model.predict(RB_ratio.reshape(-1, 1)))
    r2_GB = r2_score(Conc, GB_Model.predict(GB_ratio.reshape(-1, 1)))
    r2_RG = r2_score(Conc, RG_Model.predict(RG_ratio.reshape(-1, 1)))

    # Choose the model with the highest R-squared score
    best_model = None
    x_values = None
    y_label = None
    if r2_RB >= r2_GB and r2_RB >= r2_RG:
        best_model = RB_Model
        x_values = RB_ratio
        y_label = 'R/B ratio'
    elif r2_GB >= r2_RB and r2_GB >= r2_RG:
        best_model = GB_Model
        x_values = GB_ratio
        y_label = 'G/B ratio'
    else:
        best_model = RG_Model
        x_values = RG_ratio
        y_label = 'R/G ratio'

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

def get_analysis(RB_ratio, Conc):
    # Create pairs of elements from both arrays
    pairs = list(zip(Conc, RB_ratio))

    # Sort the pairs based on the first element of each pair
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    # Extract the second elements from the sorted pairs
    RB_ratio = np.array([pair[1] for pair in sorted_pairs])
    Conc_np = np.array([pair[0] for pair in sorted_pairs])

    return RB_ratio, Conc_np

# if __name__ == '__main__':
#     app.run()
