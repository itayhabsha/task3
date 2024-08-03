import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

def prepare_data(df):
    df = df.drop_duplicates()
    def convert_to_numeric(df):
        df['Year'] = pd.to_datetime(df['Year'], format='%Y').dt.year
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
        return df
    df = convert_to_numeric(df)

    def clean_km_column(df):
        df['Km'] = df['Km'].astype(str).str.replace(',', '').str.strip()
        df['Km'] = pd.to_numeric(df['Km'], errors='coerce')
        return df
    df = clean_km_column(df)

    def fill_null_km_with_mean(df):
        mean_km_model_year = df.groupby(['model', 'Year'])['Km'].mean().reset_index()
        mean_km_model_year.rename(columns={'Km': 'mean_Km'}, inplace=True)
        df = df.merge(mean_km_model_year, on=['model', 'Year'], how='left')
        df['Km'].fillna(df['mean_Km'], inplace=True)
        df.drop(columns=['mean_Km'], inplace=True)

        mean_km_year = df.groupby(['Year'])['Km'].mean().reset_index()
        mean_km_year.rename(columns={'Km': 'mean_Km'}, inplace=True)
        df = df.merge(mean_km_year, on=['Year'], how='left')
        df['Km'].fillna(df['mean_Km'], inplace=True)
        df.drop(columns=['mean_Km'], inplace=True)

        overall_mean_km = df['Km'].mean()
        df['Km'].fillna(overall_mean_km, inplace=True)
        df['Km'] = df['Km'].astype(int)
        return df
    df = fill_null_km_with_mean(df)

    def replace_invalid_gear_values(df):
        allowed_gear_values = ['אוטומטית', 'טיפטרוניק', 'ידנית', 'רובוטית', 'לא מוגדר']
        def replace_gear_value(row, allowed_values, df):
            value = row['Gear']
            manufactor = row['manufactor']
            if pd.isna(value) or value not in allowed_values:
                most_common_gear = df[df['manufactor'] == manufactor]['Gear']
                most_common_gear = most_common_gear[most_common_gear.isin(allowed_values)].mode()
                if not most_common_gear.empty:
                    return most_common_gear.iloc[0]
                return 'לא מוגדר'
            else:
                return value
        df['Gear'] = df.apply(lambda row: replace_gear_value(row, allowed_gear_values, df), axis=1)
        return df
    df = replace_invalid_gear_values(df)

    def clean_and_fill_capacity_engine(df):
        df['capacity_Engine'] = df['capacity_Engine'].astype(str).str.replace(',', '')
        df['capacity_Engine'] = pd.to_numeric(df['capacity_Engine'], errors='coerce')
        df.loc[df['capacity_Engine'] < 800, 'capacity_Engine'] = np.nan
        df.loc[df['capacity_Engine'] > 6000, 'capacity_Engine'] = np.nan
        df['capacity_Engine'] = df.groupby('model')['capacity_Engine'].transform(lambda x: x.fillna(x.median()))
        median_capacity = df['capacity_Engine'].median()
        df['capacity_Engine'].fillna(median_capacity, inplace=True)
        df['capacity_Engine'] = df['capacity_Engine'].astype(int)
        return df
    df = clean_and_fill_capacity_engine(df)

    def fill_missing_color(df):
        df['Color'].fillna('לא מוגדר', inplace=True)
        return df
    df = fill_missing_color(df)

    def fill_invalid_engine_type(df):
        allowed_engine_values = ['בנזין', 'דיזל', 'גז', 'היבריד', 'חשמלי']
        df['Engine_type'] = df['Engine_type'].replace('היברידי', 'היבריד')
        df['Engine_type'] = df['Engine_type'].replace('טורבו דיזל', 'דיזל')

        unique_models = df['model'].value_counts()
        single_appearance_models = unique_models[unique_models == 1].index
        df = df[~(df['model'].isin(single_appearance_models) & df['Engine_type'].isnull())].copy()

        for model in df['model'].unique():
            model_filter = df['model'] == model
            engine_type_mode = df.loc[model_filter & df['Engine_type'].isin(allowed_engine_values), 'Engine_type'].mode()
            if not engine_type_mode.empty:
                most_common_engine_type = engine_type_mode[0]
                df.loc[model_filter & (df['Engine_type'].isnull() | ~df['Engine_type'].isin(allowed_engine_values)), 'Engine_type'] = most_common_engine_type
        
        return df
    df = fill_invalid_engine_type(df)

    def convert_pic_num(df):
        def pic_convert_to_zero(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        df['Pic_num'] = df['Pic_num'].apply(pic_convert_to_zero)
        df['Pic_num'] = df['Pic_num'].astype(int)
        return df
    df = convert_pic_num(df)

    def convert_serial_dates(df, column):
        def excel_date_to_datetime(excel_date):
            try:
                epoch_start = datetime(1900, 1, 1)
                delta_days = timedelta(days=int(excel_date))
                return epoch_start + delta_days
            except ValueError:
                return pd.to_datetime(excel_date, errors='coerce')

        def is_not_date(value):
            try:
                pd.to_datetime(value, format='%d/%m/%Y', errors='raise')
                return False
            except ValueError:
                return True

        df['Non_Date_Values'] = df[column].apply(lambda x: x if is_not_date(x) else '')
        df['Converted_Date_Values'] = df[column].apply(lambda x: excel_date_to_datetime(x) if is_not_date(x) else pd.to_datetime(x, format='%d/%m/%Y'))
        df[column] = df['Converted_Date_Values']
        df.drop(columns=['Non_Date_Values', 'Converted_Date_Values'], inplace=True)
        return df
    df = convert_serial_dates(df, 'Cre_date')
    df = convert_serial_dates(df, 'Repub_date')

    def fill_ownership_columns(df, columns, default_value='לא מוגדר'):
        for column in columns:
            df[column].replace({None: default_value, 'None': default_value}, inplace=True)
            df[column].fillna(default_value, inplace=True)
        return df
    df = fill_ownership_columns(df, ['Prev_ownership', 'Curr_ownership'])

    def feature_engineering(df):
        current_date = datetime.now()
        def calculate_months_difference(start_date, end_date):
            return (end_date.year - start_date.year) * 12 + end_date.month - start_date.month
        df['months_since_Cre_date'] = df['Cre_date'].apply(lambda x: calculate_months_difference(x, current_date))
        df['Is_Repub'] = (df['Repub_date'] == df['Cre_date']).astype(int)
        return df
    df = feature_engineering(df)

    df.drop(columns=['Area', 'City', 'Description', 'Supply_score', 'Test', 'Cre_date', 'Repub_date'], inplace=True)

    return df