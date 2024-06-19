import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


file_path = 'merkez.xls'
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, xls.sheet_names[0])


st.title("Mahalle Nüfus Analizi")

mahalleler = df['Mahalle'].unique()
selected_mahalle = st.selectbox("Mahalle Seçin:", mahalleler)


mahalle_df = df[df['Mahalle'] == selected_mahalle]

# Calculate annual growth rates
mahalle_df['Nüfus Artış Oranı (%)'] = mahalle_df['nufus'].pct_change() * 100

# Layout
st.subheader(f"{selected_mahalle} Mahallesinin Nüfus Grafiği ve Analizi")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Nüfus Grafiği (2007-2023)")

    fig1, ax1 = plt.subplots()
    ax1.plot(mahalle_df['yil'], mahalle_df['nufus'], marker='o', label='Gerçek Nüfus')
    ax1.set_xlabel("Yıl")
    ax1.set_ylabel("Nüfus")
    st.pyplot(fig1)


    X = mahalle_df['yil'].values.reshape(-1, 1)
    y = mahalle_df['nufus'].values

    model = LinearRegression()
    model.fit(X, y)


    future_years = np.arange(2024, 2031).reshape(-1, 1)
    predictions = model.predict(future_years)

    st.subheader("Nüfus Tahmin Grafiği (2024-2030)")
    # Plot the prediction
    fig2, ax2 = plt.subplots()
    ax2.plot(mahalle_df['yil'], mahalle_df['nufus'], marker='o', label='Gerçek Nüfus')
    ax2.plot(future_years, predictions, marker='x', linestyle='--', color='r', label='Tahmin Edilen Nüfus')
    ax2.set_xlabel("Yıl")
    ax2.set_ylabel("Nüfus")
    ax2.legend()
    st.pyplot(fig2)

with col2:
    st.subheader("Nüfus Artış Oranı (%)")
    st.table(mahalle_df[['yil', 'Nüfus Artış Oranı (%)']])

# Reasons for population growth and decline
st.subheader("Nüfus Artış ve Azalış Nedenleri")

st.markdown("""
- **Nüfus Artış Nedenleri:**
  - Ekonomik fırsatlar ve iş olanaklarının artması
  - Göçmenlerin ve yeni yerleşimcilerin bölgeye gelmesi
  - Doğum oranlarının yüksek olması
  - Altyapı ve yaşam kalitesinin iyileşmesi

- **Nüfus Azalış Nedenleri:**
  - Ekonomik durgunluk ve işsizlik oranlarının artması
  - Göçlerin başka bölgelere yönelmesi
  - Doğum oranlarının düşmesi
  - Altyapı ve yaşam kalitesinin azalması
  - Doğal afetler ve çevresel faktörler
""")
