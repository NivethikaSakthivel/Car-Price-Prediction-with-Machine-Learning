# Car Price Prediction with Machine Learning

## Project Overview
This project predicts car prices based on various features using machine learning models. The dataset includes car details such as year, present price, driven kilometers, fuel type, selling type, transmission, and ownership status.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn

## Dataset
The dataset includes the following features:
- **Car_Name:** Name of the car
- **Year:** Year of manufacture
- **Selling_Price:** Price at which the car is sold (Target)
- **Present_Price:** Current ex-showroom price
- **Driven_kms:** Total kilometers driven
- **Fuel_Type:** Type of fuel (Petrol/Diesel)
- **Selling_type:** Type of seller (Individual/Dealer)
- **Transmission:** Manual/Automatic
- **Owner:** Number of previous owners

## Model Training and Evaluation
- **Model Used:** Random Forest Regressor
- **Performance Metrics:**
  - Mean Squared Error (MSE): 0.75
  - Root Mean Squared Error (RMSE): 0.87
  - R² Score: 0.97

## How to Run
1. Clone the repository.
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script:
   ```bash
   python script.py
   ```

## Output
- Console output displaying evaluation metrics.
- A saved plot named **actual_vs_predicted.png** showing actual vs predicted car prices.
