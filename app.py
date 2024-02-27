import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

def train_model():
    df_condo_final = pd.read_csv('cleaned_housing.csv')
    X_final = df_condo_final.drop('Price (PHP)',axis=1)
    y_final = df_condo_final['Price (PHP)']

    price_bins_final = pd.qcut(df_condo_final['Price (PHP)'], q=3, labels=['low', 'medium','high'])
    X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_final,
                                                                                y_final,
                                                                        stratify=price_bins_final,
                                                                                test_size=0.2,
                                                                                random_state=13)
    gbr_final = GradientBoostingRegressor(learning_rate=0.2,
                                          n_estimators=400, 
                                          random_state=13)
    gbr_final.fit(X_train_final, y_train_final)
    
def get_all_cities():
    feature_df = pd.read_csv('feature_df.csv')
    city_list = feature_df['city'].unique().tolist()
    return city_list

def get_brgy_for_city(city_name):
    feature_df = pd.read_csv('feature_df.csv')
    brgy_list = feature_df[feature_df['city']==city_name]['brgy'].unique().tolist()
    return brgy_list

def get_features(bedrooms, bath, fa, barangay, city):
    feature_df = pd.read_csv('feature_df.csv')
    # feature_df
    filtered_df = feature_df.query(f"`brgy`=='{barangay}' & `city`=='{city}'").copy()
    filtered_df['Bedrooms'] = bedrooms
    filtered_df['Bath'] = bath
    filtered_df['Floor_area (sqm)'] = fa
    # filtered_df['Price (PHP)'] = ""
    filtered_df["city_pop"] = filtered_df["city_pop"].str.replace(",", "")
    filtered_df = filtered_df.apply(pd.to_numeric, errors="coerce")
    
    final_columns = [
        'Latitude', 'Longitude', 'Bedrooms', 'Bath',
        'Floor_area (sqm)', 'brgy_area_sqm', 'city_pop',
        'Food_Count', 'Education_Count', 'Healthcare_Count',
        'Public_Services_Count', 'Finance_Count', 'Transportation_Count',
        # 'Price (PHP)'
    ]
    return filtered_df[final_columns]

def get_price(sample_1, model):
    sample_1['Price (PHP)'] = (model.predict(sample_1))[0]*1.4
    return sample_1

# Load and train your model at startup to avoid retraining on every interaction
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_and_train_model():
    train_model()
    # Load your trained model here instead of returning None
    # For demonstration, we assume your model is saved and loaded like this
    # return joblib.load('path_to_your_saved_model.pkl')
    return None  # Placeholder: replace with actual model loading

# Initialize your app with the trained model
model = load_and_train_model()

# Streamlit app layout
def main():
    st.title('Condo Price Prediction App')

    # Inputs
    fa = st.number_input('Floor Area', format="%.2f")
    bedrooms = st.number_input('Number of Bedrooms', step=1, format="%d")
    bath = st.number_input('Number of Bathrooms', step=1, format="%d")
    city = st.selectbox('City', get_all_cities())
    brgy = st.selectbox('Barangay', get_brgy_for_city(city))

    # Button to make prediction
    if st.button('Predict Price'):
        # Ensure the model is loaded
        if model is not None:
            features = get_features(bedrooms, bath, fa, brgy, city)
            prediction = get_price(features, model)
            st.write(f"Predicted Price (PHP): {prediction['Price (PHP)'].iloc[0]:,.2f}")
        else:
            st.error("Model is not loaded properly.")

if __name__ == "__main__":
    main()
