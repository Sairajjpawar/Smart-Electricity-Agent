# -*- coding: utf-8 -*-


import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import numpy as np
import json # For handling JSON from API
import time # For simulating delay in GPS fetch

# --- Configuration ---
# IMPORTANT: REPLACE THIS WITH YOUR ACTUAL OpenWeatherMap API KEY
# You obtained this from https://openweathermap.org/ after signing up.
# This API key is used for getting city coordinates (latitude/longitude).
# Note: Even with a valid key, OpenWeatherMap's free tier does not provide
# extensive historical weather data or solar irradiance data directly.
# The script uses simulated seasonal weather for predictions.
OPENWEATHER_API_KEY = "YourAPI"
GEOCODING_URL = "http://api.openweathermap.org/geo/1.0/direct" # To get lat/lon from city name

# --- Utility Functions for User Input ---
def get_float_input(prompt):
    """Safely gets a float input from the user."""
    while True:
        try:
            value = float(input(prompt))
            if value < 0:
                print("Input cannot be negative. Please try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_int_input(prompt, min_val=None, max_val=None):
    """Safely gets an integer input from the user with optional min/max validation."""
    while True:
        try:
            value = int(input(prompt))
            if min_val is not None and value < min_val:
                print(f"Input must be at least {min_val}. Please try again.")
            elif max_val is not None and value > max_val:
                print(f"Input cannot exceed {max_val}. Please try again.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a whole number.")

def get_city_input(prompt):
    """Gets a non-empty city name from the user."""
    while True:
        city = input(prompt).strip()
        if city:
            return city
        else:
            print("City name cannot be empty. Please enter a valid city.")

def get_monthly_data(data_type, num_months):
    """Gathers monthly data from the user."""
    data = []
    print(f"\nPlease enter the total {data_type} for the last {num_months} months:")
    current_month_start = datetime.date.today().replace(day=1)
    for i in range(num_months):
        # Calculate month in reverse order from the current month
        month_to_ask = current_month_start - relativedelta(months=(num_months - 1 - i))
        month_name = month_to_ask.strftime("%B %Y")
        value = get_float_input(f"  {month_name} {data_type} (in kWh): ")
        data.append({
            'month': month_to_ask,
            'value': value,
        })
    return data

# --- Location and Weather API Functions ---
def get_gps_location():
    """
    Placeholder for obtaining GPS location (latitude, longitude).
    In a real web/mobile application, this would involve:
    1. Client-side (web/mobile) code attempting to get GPS coordinates.
    2. Sending these coordinates to this Python backend.

    For this standalone Python script running on a desktop, direct access to
    device GPS is not possible. This function simulates the attempt and
    will always return None, prompting for manual input.
    """
    print("\nAttempting to fetch location using (simulated) device GPS...")
    # Simulate a brief delay for realism
    time.sleep(1) 
    print(" (Simulated) GPS location not found or permission denied.")
    return None, None # Simulate failure to get GPS

def get_lat_lon(city_name):
    """Gets latitude and longitude for a city using OpenWeatherMap Geocoding API."""
    params = {
        'q': city_name,
        'limit': 1,
        'appid': OPENWEATHER_API_KEY
    }
    try:
        response = requests.get(GEOCODING_URL, params=params)
        response.raise_for_status() # Raise an exception for HTTP errors (e.g., 401 for invalid API key)
        data = response.json()
        if data:
            return data[0]['lat'], data[0]['lon']
        else:
            print(f"Error: Could not find coordinates for '{city_name}'. Please check the city name spelling.")
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching geocoding data for '{city_name}': {e}")
        if "401 Client Error: Unauthorized" in str(e):
            print("Please double-check your OpenWeatherMap API key. It might be incorrect or not yet active.")
        return None, None

def get_simulated_weather_summary(month_year_date):
    """
    Provides **simulated** historical weather data (temperature, sun hours) based on
    general Indian seasonal assumptions.
    This is a proxy for actual weather influences, as granular historical weather
    or solar irradiance data is typically not available on OpenWeatherMap's free tier.
    For more accurate predictions, a paid API or more detailed data source is needed.
    """
    month_num = month_year_date.month
    
    # These values are *not* fetched from API for the free tier, but simulate influence.
    # Adjust these values if you have more specific knowledge of your city's climate
    # to better reflect typical seasonal conditions.
    temp_c = 25 # Default average temperature
    sun_hours = 8 # Default average sun hours per day

    if month_num in [3, 4, 5]: # Summer (March, April, May)
        temp_c = 32
        sun_hours = 9 # Generally sunny, but high temps can slightly reduce panel efficiency
    elif month_num in [6, 7, 8, 9]: # Monsoon (June, July, August, September)
        temp_c = 28
        sun_hours = 4 # Significantly less sun due to clouds, rain
    elif month_num in [10, 11]: # Autumn (October, November)
        temp_c = 27
        sun_hours = 7
    elif month_num in [12, 1, 2]: # Winter (December, January, February)
        temp_c = 20
        sun_hours = 8 # Clear skies, good sun, cooler temps can increase efficiency
    
    return {'avg_temp_c': temp_c, 'avg_sun_hours_per_day': sun_hours}


# --- Prediction Logic ---
def train_consumption_model(consumption_df):
    """
    Trains a simple Linear Regression model for consumption.
    Features used: Month (cyclic, for seasonality) and a simulated temperature (proxy for seasonal demand).
    With only 3-5 months of data, this model will be very basic and prone to overfitting.
    More data (daily/hourly over a year+) would enable more complex and accurate models.
    """
    if len(consumption_df) < 2:
        print("Warning: Not enough data points (need at least 2) to train a Linear Regression model for consumption.")
        print("         The agent will use the average of your provided historical consumption for predictions.")
        return None

    # Create features:
    consumption_df['month_num'] = consumption_df['month'].dt.month
    
    # Map month number to a simulated 'temperature' feature based on general seasons
    # This acts as a proxy for how temperature might influence consumption.
    temp_map_for_model = {
        1: 20, 2: 22, 3: 28, 4: 32, 5: 33, 6: 29, 7: 28, 8: 28, 9: 29, 10: 27, 11: 24, 12: 21
    }
    consumption_df['simulated_temp'] = consumption_df['month_num'].map(temp_map_for_model)

    # Convert month to cyclic features (sine and cosine) to capture yearly seasonality
    # This helps the model understand that January follows December, etc.
    consumption_df['month_sin'] = np.sin(2 * np.pi * consumption_df['month_num'] / 12)
    consumption_df['month_cos'] = np.cos(2 * np.pi * consumption_df['month_num'] / 12)

    X = consumption_df[['month_sin', 'month_cos', 'simulated_temp']]
    y = consumption_df['value']

    model = LinearRegression()
    try:
        model.fit(X, y)
        print("Consumption model trained (simple Linear Regression with seasonal features).")
        return model
    except Exception as e:
        print(f"Error training consumption model: {e}. Falling back to simple average for predictions.")
        return None


def predict_consumption(model, target_month_date, consumption_fallback_value):
    """
    Predicts monthly consumption using the trained model or a fallback average.
    """
    if model is None:
        return consumption_fallback_value # Use simple average if model couldn't be trained

    pred_month_num = target_month_date.month
    
    # Get simulated temperature for the prediction month, consistent with training data
    temp_map_for_model = {
        1: 20, 2: 22, 3: 28, 4: 32, 5: 33, 6: 29, 7: 28, 8: 28, 9: 29, 10: 27, 11: 24, 12: 21
    }
    pred_simulated_temp = temp_map_for_model.get(pred_month_num, 25) # Default if somehow not found

    # Create features for prediction consistent with training features
    pred_month_sin = np.sin(2 * np.pi * pred_month_num / 12)
    pred_month_cos = np.cos(2 * np.pi * pred_month_num / 12)
    
    X_pred = pd.DataFrame([[pred_month_sin, pred_month_cos, pred_simulated_temp]],
                          columns=['month_sin', 'month_cos', 'simulated_temp'])
    
    predicted_kwh = model.predict(X_pred)[0]
    return max(0, predicted_kwh) # Ensure prediction is not negative (consumption cannot be negative)

def predict_solar_generation(solar_kwp, target_month_date):
    """
    Predicts monthly solar generation based on kWp and estimated sun hours for the month.
    Uses a common rule of thumb for India: 1 kWp installed capacity generates ~4-5 kWh per day
    under optimal conditions. This base is adjusted by simulated sun hours to reflect seasonality.
    """
    kwh_per_kwp_per_day_base = 4.0 # A conservative base average kWh generated per day per 1 kWp

    # Get simulated sun hours for the prediction month from our internal logic
    weather_summary = get_simulated_weather_summary(target_month_date)
    avg_sun_hours_per_day = weather_summary['avg_sun_hours_per_day']

    # Calculate days in the target month accurately
    days_in_month = (target_month_date + relativedelta(months=1) - target_month_date).days
    
    # Adjust base generation rate by sun hours (simplified ratio)
    # 8.0 is used as a reference for "average" good sun hours per day.
    generation_factor_from_sun = avg_sun_hours_per_day / 8.0
    
    # Cap the factor to prevent overly optimistic/pessimistic predictions
    # This helps in making the predictions more "reliable" by preventing extreme outliers.
    if generation_factor_from_sun > 1.2:
        generation_factor_from_sun = 1.2
    elif generation_factor_from_sun < 0.3:
        generation_factor_from_sun = 0.3

    predicted_kwh = solar_kwp * kwh_per_kwp_per_day_base * days_in_month * generation_factor_from_sun
    return max(0, predicted_kwh) # Ensure prediction is not negative (generation cannot be negative)

def main():
    print("Welcome to the AI Smart Agent for Electricity Management!")
    print("Let's gather some information to predict your energy future.")
    print("This prototype will provide an estimation. For higher accuracy, more data and advanced features are needed.")

    # --- API Key Check ---
    if OPENWEATHER_API_KEY == "YOUR_OPENWEATHERMAP_API_KEY" or not OPENWEATHER_API_KEY.strip():
        print("\n========================================================================")
        print(" CRITICAL ERROR: OpenWeatherMap API Key is not configured correctly. ")
        print(" Please replace 'YOUR_OPENWEATHERMAP_API_KEY' in the script with your actual key.")
        print(" Get your free API key after signing up at: https://openweathermap.org/api")
        print(" Your API key may take a few minutes to become active after creation.")
        print("========================================================================")
        return

    # --- 1. Get Consumption Data ---
    num_consumption_months = get_int_input(
        "How many previous months of electricity consumption data (total monthly kWh) do you have (3-5 months recommended)? ",
        min_val=3, max_val=5
    )
    past_consumption_data = get_monthly_data("electricity consumption", num_consumption_months)
    consumption_df = pd.DataFrame(past_consumption_data)
    
    # --- IMPORTANT FIX: Ensure the 'month' column is explicitly converted to datetime objects ---
    # This is crucial for .dt accessor to work correctly for extracting month_num etc.
    consumption_df['month'] = pd.to_datetime(consumption_df['month'])
    
    # Calculate average consumption for fallback if model training fails
    average_historical_consumption = consumption_df['value'].mean() if not consumption_df.empty else 0

    # --- 2. Train Consumption Model ---
    consumption_model = train_consumption_model(consumption_df.copy()) # Pass a copy to avoid modifying original df for other uses


    # --- 3. Get Solar Panel Data ---
    solar_panel_kwp = get_float_input(
        "\nEnter the rated capacity of your solar panels in kWp (e.g., 5.0 for 5 kWp): "
    )

    # --- 4. Get City and Prediction Horizon ---
    city_lat, city_lon = None, None
    city_name = ""

    # Attempt to get location via (simulated) GPS first
    # In a real web/mobile app, client-side code would pass lat/lon here.
    # For this desktop Python script, get_gps_location() will always return None.
    gps_lat, gps_lon = get_gps_location()
    
    if gps_lat is not None and gps_lon is not None:
        # This block would be used if real GPS data was passed (e.g., from a web app)
        print("Location fetched successfully via (simulated) GPS!")
        # In a real scenario, you would use a reverse geocoding API here
        # to get a city name from the lat/lon. For this prototype, we'll
        # just use a placeholder city name for the GPS-derived location.
        city_name = "Detected City (e.g., Bengaluru)" # Placeholder
        city_lat, city_lon = gps_lat, gps_lon
        print(f"Using location: {city_name} (Lat: {city_lat:.2f}, Lon: {city_lon:.2f})")
    else:
        # Fallback: If GPS failed, ask the user for the city name
        print("Falling back to manual city input.")
        city_name = get_city_input("Which Indian city are you in (e.g., Mumbai, Delhi, Bengaluru)? ")
        city_lat, city_lon = get_lat_lon(city_name) # Use OpenWeatherMap to get coords for the manually entered city


    if city_lat is None or city_lon is None:
        print("\nPrediction cancelled due to invalid city or API key issue. Please fix the error and try again.")
        return

    num_prediction_months = get_int_input(
        f"How many upcoming months do you want to predict for {city_name} (e.g., 3, 6, 12)? ",
        min_val=1
    )

    print(f"\n--- Generating Predictions for {city_name} for the next {num_prediction_months} months ---")

    predictions_results = []
    # Start prediction from the beginning of the next month, relative to current date.
    # This ensures we predict truly "future" months.
    current_date_for_prediction = datetime.date.today().replace(day=1) + relativedelta(months=1)


    for i in range(num_prediction_months):
        target_month_date = current_date_for_prediction + relativedelta(months=i)
        month_label = target_month_date.strftime("%B %Y")
        
        # Predict Consumption
        predicted_consumption = predict_consumption(consumption_model, target_month_date, average_historical_consumption)

        # Predict Solar Generation
        predicted_generation = predict_solar_generation(solar_panel_kwp, target_month_date)

        # Calculate Surplus/Deficit
        surplus_deficit = predicted_generation - predicted_consumption

        predictions_results.append({
            "month": month_label,
            "predicted_consumption": predicted_consumption,
            "predicted_generation": predicted_generation,
            "surplus_deficit": surplus_deficit
        })

    # --- Output Summary ---
    print("\n--- Energy Balance Prediction Summary ---")
    for p in predictions_results:
        status = "Surplus" if p["surplus_deficit"] >= 0 else "Deficit"
        print(f"\nMonth: {p['month']}")
        print(f"  Predicted Consumption: {p['predicted_consumption']:.2f} kWh")
        print(f"  Predicted Solar Generation: {p['predicted_generation']:.2f} kWh")
        print(f"  Overall: {status} of {abs(p['surplus_deficit']):.2f} kWh")
        if status == "Surplus":
            print("  Action Suggestion: You are predicted to generate more electricity than you consume. Consider storing surplus or feeding to grid.")
        else:
            print("  Action Suggestion: You are predicted to consume more electricity than you generate. Consider reducing consumption, increasing solar capacity, or alternative power sources.")

    print("\n--- End of Prediction Report ---")
    print("\nTo improve accuracy and make the agent truly 'smart':")
    print("1. Provide more historical consumption data (daily/hourly over a year or more).")
    print("2. Collect actual historical solar generation data from your inverter.")
    print("3. Explore paid weather APIs for precise historical solar irradiance data.")
    print("4. Consider more advanced machine learning models (e.g., Prophet, ARIMA, Gradient Boosting) with more data.")
    print("\nThis prototype is a starting point for your electricity management journey!")

if __name__ == "__main__":
    main()

