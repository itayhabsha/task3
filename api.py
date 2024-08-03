import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
from car_data_prep import prepare_data
import model_training

app = Flask(__name__)
model = pickle.load(open('trained_model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = request.form.getlist('feature')
    
    # Creating a DataFrame with the entered features
    df_one = pd.DataFrame([features], columns=[
        'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine',
        'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Cre_date', 'Repub_date',
        'Pic_num', 'Color', 'Km'
    ])
    
    # Adding new columns with a default value of 0
    df_one['Area'] = 0
    df_one['City'] = 0
    df_one['Description'] = 0
    df_one['Test'] = 0
    df_one['Supply_score'] = 0
    df_one['Price'] = 0
    
    # Replacing missing and infinite values with 0
    df_one.replace([np.nan], 0, inplace=True)
    
    # Converting columns to integers
    df_one['Km'] = df_one['Km'].astype(int)
    df_one['Area'] = df_one['Area'].astype(int)
    df_one['City'] = df_one['City'].astype(int)
    df_one['Description'] = df_one['Description'].astype(int)
    df_one['Test'] = df_one['Test'].astype(int)
    df_one['Supply_score'] = df_one['Supply_score'].astype(int)
    df_one['Price'] = df_one['Price'].astype(int)
    
    ordered_columns = [
        'manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 
        'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 
        'Price', 'Pic_num', 'Cre_date', 'Repub_date', 'Description', 'Color', 
        'Km', 'Test', 'Supply_score'
    ]
    
    df_one = df_one.reindex(columns=ordered_columns, fill_value=0)
    
    # Preparing the data
    df_one = prepare_data(df_one)
    df_one.drop(columns=['Price'], inplace=True)
    
    # One-Hot Encoding of categorical variables
    cols_to_encode = ['Gear', 'Engine_type', 'model', 'Prev_ownership', 'Curr_ownership', 'manufactor', 'Color']
    encoded_columns = encoder.transform(df_one[cols_to_encode])
    
    # Creating DataFrame for encoded columns
    encoded_column_names = encoder.get_feature_names_out(cols_to_encode)
    encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names)
    
    # Dropping original categorical columns and combining encoded columns
    df_one.drop(columns=cols_to_encode, inplace=True)
    df_one = pd.concat([df_one.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    
    # Standardizing selected columns
    columns_to_standardize = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Pic_num', 'months_since_Cre_date']
    df_one[columns_to_standardize] = scaler.transform(df_one[columns_to_standardize])
    
    # Aligning features with training data
    X_train = model_training.X_train
    missing_cols = set(X_train.columns) - set(df_one.columns)
    for col in missing_cols:
        df_one[col] = 0
    df_one = df_one[X_train.columns]
    
    # Prediction
    predicted_price = model.predict(df_one)
    print(f"Predicted Price: {predicted_price[0]}")
    
    return render_template('index.html', prediction_text='{}'.format(predicted_price))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
