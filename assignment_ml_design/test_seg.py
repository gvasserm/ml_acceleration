import json
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import vpr_toolbox.utils.data_utils as data_utils
from tqdm import tqdm

def add2mask(mask, labelID, polygon_points):

    # Reshape the points to fit OpenCV's format
    polygon_points = polygon_points.reshape((-1, 1, 2))

    # Draw the polygon on the mask
    cv2.fillPoly(mask, [polygon_points], labelID)
    return mask


def plot_masked(image, mask, labelID, pallet):

    # Define transparency factor
    alpha = 0.5  # Transparency factor (0: completely transparent, 1: completely opaque)

    # Create a color version of the mask (red color)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == labelID] = pallet  # Red color for the mask

    # Blend the original image with the mask
    blended_image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)
    return blended_image

import csv
def create_lists():

    #img = cv2.imread("/home/gvasserm/data/cityscapes_tiny/gtFine/val/munster/munster_000000_000019_gtFine_labelIds.png")
    
    # gt = cv2.imread("/home/gvasserm/data/IndianRoads/train/gt/1_012_frame0000000623_leftImg8bit.png")
    # img = cv2.imread("/home/gvasserm/data/IndianRoads/train/img/1_012_frame0000000623_leftImg8bit.jpg")
    # plt.imshow(gt)
    # plt.show()

    split = "train"

    annotations_folder = f"/home/gvasserm/data/IndianRoads/{split}/ann/"
    image_folder = f"{split}/img/"
    gt_folder = f"{split}/gt/"

    database_paths = data_utils.find_files_in_dir(annotations_folder, extensions=["*.json"])


    data = []
    for fname in tqdm(database_paths):
        imname = image_folder + fname.split('/')[-1].split('.json')[0]
        gt_name = gt_folder + fname.split('/')[-1].split('.jpg')[0] + '.png'

        data.append([imname, gt_name])

    
     # File path
    file_path = f'/home/gvasserm/data/IndianRoads/{split}.lst'

    # Writing to CSV file
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=' ')
        # Writing the data
        writer.writerows(data)

        print(f"CSV file '{file_path}' created successfully.")
    return


def convert2masks():

    labels = {"road": 1, "footpath": 2, "shallow": 3, "pothole": 4, "background": 0}

    split = "val"

    annotations_folder = f"/home/gvasserm/data/IndianRoads/{split}/ann/"
    image_folder = f"/home/gvasserm/data/IndianRoads/{split}/img/"
    gt_folder = f"/home/gvasserm/data/IndianRoads/{split}/gt/"

    database_paths = data_utils.find_files_in_dir(annotations_folder, extensions=["*.json"])

    for fname in tqdm(database_paths):
        with open(fname, "r") as f:
            data = json.load(f)

        sample_img = cv2.imread(image_folder + fname.split('/')[-1].split('.json')[0])
        mask = np.zeros((sample_img.shape[0], sample_img.shape[1]), dtype=np.uint8)
        
        pallet = [[255,0,0],[255,0,255],[0,255,255],[0,0,255]]

        polygons = []

        for i in range(len(data['objects'])):    
            polygons.append(np.asarray(data['objects'][i]['points']['exterior'], np.int32))

            l = data['objects'][i]['classTitle']

            mask = add2mask(mask, labels[l], polygons[-1])

            #sample_img = plot_masked(sample_img, mask, labels[l], pallet[labels[l]])
            #plt.plot(p[:,0], p[:, 1], '-b.')

        if np.max(mask) >= 5:
            print("BUG")

        cv2.imwrite(gt_folder + fname.split('/')[-1].split('.jpg')[0] + '.png', mask)
        
        #plt.imshow(sample_img)
        #out = from_ann_to_cityscapes_mask(data, name2id, train_val_flag)

        # for p in polygons:
        #     plt.plot(p[:,0], p[:, 1], '-b.')
        # plt.show()
    return

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def ex3():
    # Documents
    documents = [
        "Как купить полис ОСАГО?",
        "Не пришел полис",
        "Я пришел к вам за ОСАГО",
        "Как взять не займ, а кредит?",
        "Как взять ОСАГО?"
    ]

    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute the cosine similarity between document 5 and all other documents
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()

    # Create a dataframe to hold the results
    df = pd.DataFrame({
        'Document': [1, 2, 3, 4],
        'Cosine Similarity': cosine_similarities
    })

    # Sort by cosine similarity in descending order
    df = df.sort_values(by='Cosine Similarity', ascending=False)
    print(df['Document'].values)
    return

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import tensorflow as tf

from imblearn.over_sampling import SMOTE


# Define the focal loss function
def focal_loss(gamma=2., alpha=0.7):
    def focal_loss_fixed(y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred) + K.epsilon()
        fl = - alpha_t * K.pow((tf.ones_like(y_true) - p_t), gamma) * K.log(p_t)
        return K.mean(fl)
    return focal_loss_fixed

def ex7():

    dataset_path = "/home/gvasserm/Downloads/archive/dataset.csv"
    df = pd.read_csv(dataset_path)

    # Replace missing values in 'education' and 'previous_year_rating' with the most frequent value
    imputer = SimpleImputer(strategy='most_frequent')
    df[['education', 'previous_year_rating']] = imputer.fit_transform(df[['education', 'previous_year_rating']])

    # Define categorical features and numerical features
    categorical_features = ['department', 'region', 'education', 'gender', 'recruitment_channel']
    numerical_features = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'KPIs_met >80%', 'awards_won?', 'avg_training_score']

    # Encode categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    # Split the data into training and testing sets
    X = df.drop(columns=['employee_id', 'is_promoted'])
    y = df['is_promoted']
    
    # # Apply preprocessing to the features
    X = preprocessor.fit_transform(X)
    X = X.toarray()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    # Build the MLP model
    model = Sequential([
        Dense(64, input_dim=X_train.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer='adam', loss=focal_loss(), metrics=['accuracy'])

    # Train the model
    model.fit(X_train_res.astype(np.float32), y_train_res.astype(np.float32), epochs=5, batch_size=10, validation_split=0.2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test.astype(np.float32), y_test.astype(np.float32))
    #print(classification_report(y_test, y_pred))
    print(f'Test Accuracy: {accuracy:.4f}')

    y_pred = (model.predict(X_test.astype(np.float32)) > 0.5).astype(np.int32)

    # Print classification report
    print(classification_report(y_test, y_pred))

    return


#convert2masks()
#create_lists()
ex7()
    