#uppgift 15b

#importera biliotek
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.title("Bilpris-prediktion")
st.write("Fyll i information om bilen för att få ett predikterat pris.")

# Läser in datasetet
df = pd.read_csv("car_price_dataset.csv", sep=";")

# Tar bort  mellanslag i kolumnnamn
df.columns = df.columns.str.strip()

# Tar bort rader med saknade värden
df = df.dropna()

# vi delar upp i X och y
X = df.drop(columns="Price")
y = df["Price"]

# Här identifierar vi vilka kolumner som är kategoriska och numeriska
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

# Här skapar en preprocessing-del OneHotEncoder omvandlar kategoriska variabler till numerisk form och sedan skapar vi en pipe line som preprocessar datan och sedan tränas modellen
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"  
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42))
])


# Tränar modellen på all data
model.fit(X, y)


brand = st.selectbox("Välj bilmärke", sorted(df["Brand"].unique()))
model_name = st.selectbox("Välj modell", sorted(df["Model"].unique()))
year = st.number_input("Årsmodell", min_value=int(df["Year"].min()), max_value=int(df["Year"].max()), value=int(df["Year"].median()))
engine_size = st.number_input("Motorstorlek", min_value=float(df["Engine_Size"].min()), max_value=float(df["Engine_Size"].max()), value=float(df["Engine_Size"].median()))
fuel_type = st.selectbox("Bränsletyp", sorted(df["Fuel_Type"].unique()))
transmission = st.selectbox("Växellåda", sorted(df["Transmission"].unique()))
mileage = st.number_input("Miltal / körsträcka", min_value=int(df["Mileage"].min()), max_value=int(df["Mileage"].max()), value=int(df["Mileage"].median()))
doors = st.number_input("Antal dörrar", min_value=int(df["Doors"].min()), max_value=int(df["Doors"].max()), value=int(df["Doors"].median()))
owner_count = st.number_input("Antal tidigare ägare", min_value=int(df["Owner_Count"].min()), max_value=int(df["Owner_Count"].max()), value=int(df["Owner_Count"].median()))

# Skapar en DataFrame med användarens inmatning
input_data = pd.DataFrame({
    "Brand": [brand],
    "Model": [model_name],
    "Year": [year],
    "Engine_Size": [engine_size],
    "Fuel_Type": [fuel_type],
    "Transmission": [transmission],
    "Mileage": [mileage],
    "Doors": [doors],
    "Owner_Count": [owner_count]
})

# En knapp för att göra prediktion
if st.button("Prediktera pris"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predikterat pris: {prediction:,.2f}")