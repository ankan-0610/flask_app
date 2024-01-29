# python_server.py
from flask import Flask, request, jsonify
from matplotlib import pyplot as plt
import io
import base64
import cv2
import requests
import numpy as np
from sklearn.linear_model import LinearRegression

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

    LR = LinearRegression()
    LR.fit(RB_ratio.reshape(-1, 1), Conc)
    RB_pred = LR.predict(RB_ratio.reshape(-1, 1))

    plt.scatter(Conc, RB_ratio)
    plt.plot(RB_pred, RB_ratio)
    plt.xlabel("Conc in uM")
    plt.ylabel("B/R ratio")

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
    table_data = [{'Column1': 'Value1', 'Column2': 'Value2'}, {'Column1': 'Value3', 'Column2': 'Value4'}]

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
