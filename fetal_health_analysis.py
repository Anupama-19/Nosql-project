import streamlit as st
import pymongo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# MongoDB Connection
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "fetal_health"
COLLECTION_NAME = "records"

@st.cache_data
def load_data():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ObjectId
    return pd.DataFrame(data)

# Load Data
df = load_data()

# Streamlit UI
st.set_page_config(page_title="Fetal Health Analysis", layout="wide")
st.title("📊 Fetal Health Analysis")
st.write("This dashboard analyzes the correlation between maternal health parameters and fetal outcomes.")

# Show dataset
if st.checkbox("Show Dataset"):
    st.write(df.head())

# Compute Correlation
st.subheader("🔗 Correlation Analysis")
if not df.empty:
    correlation_matrix = df.corr()

    # Heatmap Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # User-Selected Correlation
    st.subheader("📈 Select Parameters to Analyze Correlation")
    x_param = st.selectbox("Select X-axis parameter", df.columns)
    y_param = st.selectbox("Select Y-axis parameter", df.columns)

    if x_param and y_param:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=df[x_param], y=df[y_param], alpha=0.7)
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"Correlation between {x_param} and {y_param}")
        st.pyplot(fig)

        # Display correlation value
        correlation_value = df[x_param].corr(df[y_param])
        st.markdown(f"### 📊 Correlation Coefficient: `{correlation_value:.2f}`")

else:
    st.warning("No data found in the database!")

# Footer
st.markdown("---")
st.markdown("Developed by **Sneha Ramesh** | Ed-Tech & Data Enthusiast 🚀")
