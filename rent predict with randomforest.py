import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("rent-data.csv") 
data.columns = data.columns.str.strip()  

# Split features and target
X = data.drop("monthly_rent_usd", axis=1)
y = data["monthly_rent_usd"]

# Define categorical and numerical columns
categorical_features = ['location', 'property_type']
numerical_features = ['area_sqft']

# Preprocessing: OneHotEncode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create the pipeline with preprocessing and model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Evaluate (optional)
y_pred = model.predict(X_test)
print(f"Model R² score: {r2_score(y_test, y_pred):.2f}")
print(f"Model Mean Squared Error: {mean_squared_error(y_test, y_pred):.2f}")

# --- USER INPUT SECTION ---
print("\n--- Predict Rent for New Property ---")
try:
    area_sqft = float(input("Enter area in square feet (e.g. 900): "))
    location = input("Enter location (e.g. Downtown Historic): ").strip()
    property_type = input("Enter property type (e.g. Studio Loft): ").strip()

    user_df = pd.DataFrame([{
        'area_sqft': area_sqft,
        'location': location,
        'property_type': property_type
    }])

    # Predict rent
    predicted_rent = model.predict(user_df)[0]
    print(f"\n🏠 Estimated Rent: ${predicted_rent:,.2f}/month")

except Exception as e:
    print("⚠️ Error:", e)
