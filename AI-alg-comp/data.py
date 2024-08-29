import numpy as np
import pandas as pd
import os
import cv2
def smoking():
    data = pd.read_csv("Resources//smoking.csv")
    data = data.drop("ID", axis=1)
    data['gender'] = [1 if sex == "F" else 0 for sex in data['gender']]
    data["oral"] = [1 if sex == "Y" else 0 for sex in data["oral"]]
    data["tartar"] = [1 if sex == "Y" else 0 for sex in data["tartar"]]
    test_idx = np.random.choice(range(data.shape[0]), round(0.8 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("smoking", axis=1).to_numpy()
    y_train = data_train["smoking"].to_numpy()
    X_test = data_test.drop("smoking", axis=1).to_numpy()
    y_test = data_test["smoking"].to_numpy()

    return X_train,y_train,X_test,y_test

def breast_cancer():
    data = pd.read_csv("Resources//breast_cancer.csv")
    data = data.drop("id", axis=1)
    data['diagnosis'] = [1 if diagnose == "M" else 0 for diagnose in data['diagnosis']]
    test_idx = np.random.choice(range(data.shape[0]), round(0.8 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("diagnosis", axis=1).to_numpy()
    y_train = data_train["diagnosis"].to_numpy()
    X_test = data_test.drop("diagnosis", axis=1).to_numpy()
    y_test = data_test["diagnosis"].to_numpy()

    return X_train,y_train,X_test,y_test

def cardio():
    data = pd.read_csv("Resources//cardio_train.csv")
    data = data.drop("id", axis=1)
    test_idx = np.random.choice(range(data.shape[0]), round(0.8 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("cardio", axis=1).to_numpy()
    y_train = data_train["cardio"].to_numpy()
    X_test = data_test.drop("cardio", axis=1).to_numpy()
    y_test = data_test["cardio"].to_numpy()

    return X_train, y_train, X_test, y_test

def car_insurance():
    data = pd.read_csv("Resources//Car_Insurance_Claim.csv")
    data = data.drop("ID", axis=1)
    data['CREDIT_SCORE'] = [1 if score == np.nan else 0 for score in data['CREDIT_SCORE']]
    data['ANNUAL_MILEAGE'] = [1 if mileage == np.nan else 0 for mileage in data['ANNUAL_MILEAGE']]
    test_idx = np.random.choice(range(data.shape[0]), round(0.8 * data.shape[0]), replace=False)
    data_test = data.iloc[test_idx, :]
    data_train = data.drop(test_idx, axis=0)
    X_train = data_train.drop("OUTCOME", axis=1).to_numpy()
    y_train = data_train["OUTCOME"].to_numpy()
    X_test = data_test.drop("OUTCOME", axis=1).to_numpy()
    y_test = data_test["OUTCOME"].to_numpy()

    return X_train,y_train,X_test,y_test

def load_images_from_folder(folder_path, image_size):
    images = []
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, image_size).flatten()
        images.append(image)
    return images

def prepare_dataset(false_path_folder, true_path_folder, image_size):
    X = []
    y = []

    false_path_images = load_images_from_folder(false_path_folder, image_size)
    X.extend(false_path_images)
    y.extend([0] * len(false_path_images))

    true_path_images = load_images_from_folder(true_path_folder, image_size)
    X.extend(true_path_images)
    y.extend([1] * len(true_path_images))

    return np.array(X), np.array(y)

def photos(false_path,true_path,size, test_size):
    image_size = (size,size)

    X, y = prepare_dataset(false_path, true_path, image_size)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train,y_train,X_test,y_test

