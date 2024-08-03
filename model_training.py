import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from car_data_prep import prepare_data

# Load the dataset
df_two = pd.read_csv('dataset.csv')



# Prepare the data
df_prepared = prepare_data(df_two)

# Data cleaning for df_le DataFrame
df_le = df_prepared.copy()
df_le = df_le.dropna()  # Remove missing values
df_le = df_le.replace([np.inf, -np.inf], np.nan).dropna()  # Remove infinite values

# Convert relevant columns to numeric explicitly
numeric_columns = ['Year', 'Hand', 'Price', 'Pic_num', 'Km', 'capacity_Engine', 'months_since_Cre_date', 'Is_Repub']
df_le[numeric_columns] = df_le[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Define the features and target variable
X = df_le.drop(columns=['Price'])  # Features
y = df_le['Price']  # Target variable

# One-Hot Encoding of categorical variables
cols_to_encode = ['Gear', 'Engine_type', 'model', 'Prev_ownership', 'Curr_ownership', 'manufactor', 'Color']
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[cols_to_encode])



# Create DataFrame for encoded columns
encoded_column_names = encoder.get_feature_names_out(cols_to_encode)
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_column_names)

# Drop original categorical columns and concatenate the encoded columns
X.drop(columns=cols_to_encode, inplace=True)
X = pd.concat([X.reset_index(drop=True), X_encoded_df.reset_index(drop=True)], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.13, random_state=80)

# Standardize specified columns
columns_to_standardize = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Pic_num', 'months_since_Cre_date']
scaler = StandardScaler()
X_train[columns_to_standardize] = scaler.fit_transform(X_train[columns_to_standardize])
X_test[columns_to_standardize] = scaler.transform(X_test[columns_to_standardize])

# Build the ElasticNet model with increased iterations
model = ElasticNet(alpha=0.1, l1_ratio=1, max_iter=50000, tol=0.01)

# Fit the model
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open("trained_model.pkl", "wb"))

with open('encoder.pkl', 'wb') as f:
    pickle.dump(encoder, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model training complete and model saved.")

# Prediction phase
