import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import requests
import cloudpickle as cp
import json
import firebase_admin
from firebase_admin import credentials, storage

# Normalize features
def normalize(arr):
    return [arr[0]/arr, arr/arr[0], (1-(arr[0]/arr)), (1-arr/arr[0]), (arr[0]-arr),(arr-arr[0]), arr]

def upload_file_to_bucket(file_path):
    bucket = storage.bucket()
    destination_blob_name = f'ml_models/{file_path}'
    blob = bucket.blob(destination_blob_name)
    # Upload the file
    blob.upload_from_filename(file_path)
    
    print(f"File {file_path} uploaded to {destination_blob_name}.")

def train_rf_model(image_urls, user_inputs):

    mean_R, mean_G, mean_B = np.array([]), np.array([]), np.array([])
    mean_S, mean_V = np.array([]), np.array([])
    mean_gray = np.array([])
    conc_values = np.array(user_inputs, dtype=float)

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
            B, G, R = cv2.split(img)

            mean_R = np.append(mean_R, np.mean(R))
            mean_G = np.append(mean_G, np.mean(G))
            mean_B = np.append(mean_B, np.mean(B))

            HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            H, S, V = cv2.split(HSV_image)
            mean_gray = np.append(mean_gray, np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            # print(mean_G)
            mean_S = np.append(mean_S, np.mean(S))
            mean_V = np.append(mean_V, np.mean(V))
            # Extract concentration value from filename
        except requests.exceptions.RequestException as e:
            print(f"Error fetching image: {e}")
            return
    
    # Stack all features into one array
    all_features = np.vstack((mean_R, mean_G, mean_B, mean_gray, mean_S, mean_V))

    # Normalize each feature and generate four new features for each original feature
    normalized_features = np.array([normalize(feature) for feature in all_features]).T

    # Reshape the normalized_features array to have only two dimensions
    normalized_features = normalized_features.reshape(normalized_features.shape[0], -1)

    feature_names = ["R", "G", "B", "Gray", "S", "V"]

    # Flatten the nested list using list comprehension and extend method
    flattened_feature_names = []
    [flattened_feature_names.extend(sublist) for sublist in 
    [[feat + "[0]/" + feat, feat + "/" + feat + "[0]", "1-" + feat + "[0]/" + feat, "1-" + feat + "/" + feat + "[0]",
    feat+"[0]-"+feat,feat+"-"+feat+"[0]",feat] for feat in feature_names]]

    # Calculate the Pearson correlation coefficients of the new normalized features with the concentrations
    corrcoef_matrix= np.corrcoef(normalized_features.T, conc_values)
    correlation_with_conc_values = corrcoef_matrix[-1, :-1]

    # Find the index of the feature with the highest absolute correlation coefficient
    selected_feature_index = np.argmax(np.abs(correlation_with_conc_values))

    # Extract the selected feature
    selected_feature = normalized_features[:,selected_feature_index]

    # Train a Random Forest Regressor model with the selected feature
    RF_model = RandomForestRegressor(n_estimators=100, random_state=42)
    RF_model.fit(selected_feature.reshape(-1, 1), conc_values)

    conc_pred_RF = RF_model.predict(selected_feature.reshape(-1, 1))

    print("\nRandom Forest Regressor Model:")
    print(f"Selected feature: {flattened_feature_names[selected_feature_index]}")
    print(f"Pearson correlation coefficient: {correlation_with_conc_values[selected_feature_index]}")
    print(f"Mean absolute error: {mae_RF}")
    print(f"R^2 score: {r2_RF}")

    with open('BR_model.pkl', 'wb') as f:
        cp.dump(RF_model, f)

    # Load the service account JSON from environment variable
    service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
    cred = credentials.Certificate(service_account_info)
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'image-poc-1ba68.appspot.com'
    })

    upload_file_to_bucket('BR_model.pkl')



mean_R, mean_G, mean_B = np.array([]), np.array([]), np.array([])
mean_S, mean_V, conc_values = np.array([]), np.array([]), np.array([])
mean_gray = np.array([])

for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        filepath = os.path.join(images_dir, filename)
        img = cv2.imread(filepath)
        B, G, R = cv2.split(img)

        mean_R = np.append(mean_R, np.mean(R))
        mean_G = np.append(mean_G, np.mean(G))
        mean_B = np.append(mean_B, np.mean(B))

        HSV_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(HSV_image)
        mean_gray = np.append(mean_gray, np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
        # print(mean_G)
        mean_S = np.append(mean_S, np.mean(S))
        mean_V = np.append(mean_V, np.mean(V))
        # Extract concentration value from filename
        concentration = float(filename[:2])
        conc_values = np.append(conc_values, concentration)

# Stack all features into one array
all_features = np.vstack((mean_R, mean_G, mean_B, mean_gray, mean_S, mean_V))

# Normalize each feature and generate four new features for each original feature
normalized_features = np.array([normalize(feature) for feature in all_features]).T

# Reshape the normalized_features array to have only two dimensions
normalized_features = normalized_features.reshape(normalized_features.shape[0], -1)

feature_names = ["R", "G", "B", "Gray", "S", "V"]

# Flatten the nested list using list comprehension and extend method
flattened_feature_names = []
[flattened_feature_names.extend(sublist) for sublist in 
[[feat + "[0]/" + feat, feat + "/" + feat + "[0]", "1-" + feat + "[0]/" + feat, "1-" + feat + "/" + feat + "[0]",
feat+"[0]-"+feat,feat+"-"+feat+"[0]",feat] for feat in feature_names]]

# Calculate the Pearson correlation coefficients of the new normalized features with the concentrations
corrcoef_matrix= np.corrcoef(normalized_features.T, conc_values)
correlation_with_conc_values = corrcoef_matrix[-1, :-1]

# Find the index of the feature with the highest absolute correlation coefficient
selected_feature_index = np.argmax(np.abs(correlation_with_conc_values))

# Extract the selected feature
selected_feature = normalized_features[:,selected_feature_index]

sorted(selected_feature)

# # Train a linear regression model with the selected feature
# LR_model1 = LinearRegression()
# LR_model1.fit(selected_feature.reshape(-1, 1)[:10], conc_values[:10])
# print("Coeff 1: ",LR_model1.coef_)

# LR_model2 = LinearRegression()
# LR_model2.fit(selected_feature.reshape(-1, 1)[10:], conc_values[10:])
# print("Coeff 2: ",LR_model2.coef_)

# Train a Random Forest Regressor model with the selected feature
RF_model = RandomForestRegressor(n_estimators=100, random_state=42)
RF_model.fit(selected_feature.reshape(-1, 1), conc_values)

# # Make predictions with both models
# conc_pred_LR1 = LR_model1.predict(selected_feature.reshape(-1, 1)[:10])
# conc_pred_LR2 = LR_model2.predict(selected_feature.reshape(-1, 1)[10:])
conc_pred_RF = RF_model.predict(selected_feature.reshape(-1, 1))

# # Evaluate the models
# mae_LR1 = mean_absolute_error(conc_values[:10], conc_pred_LR1)
# r2_LR1 = r2_score(conc_values[:10], conc_pred_LR1)

# mae_LR2 = mean_absolute_error(conc_values[10:], conc_pred_LR2)
# r2_LR2 = r2_score(conc_values[10:], conc_pred_LR2)

mae_RF = mean_absolute_error(conc_values, conc_pred_RF)
r2_RF = r2_score(conc_values, conc_pred_RF)

# # Print the results
# print("Linear Regression Model:")
# print(f"Selected feature: {flattened_feature_names[selected_feature_index]}")
# print(f"Pearson correlation coefficient: {correlation_with_conc_values[selected_feature_index]}")
# print(f"MAE1: {mae_LR1}")
# print(f"R^2 score 1: {r2_LR1}")

# print(f"MAE2: {mae_LR2}")
# print(f"R^2 score 2: {r2_LR2}")

print("\nRandom Forest Regressor Model:")
print(f"Selected feature: {flattened_feature_names[selected_feature_index]}")
print(f"Pearson correlation coefficient: {correlation_with_conc_values[selected_feature_index]}")
print(f"Mean absolute error: {mae_RF}")
print(f"R^2 score: {r2_RF}")

with open('BR_model.pkl', 'wb') as f:
    cp.dump(RF_model, f)
