import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

file_path = 'merkez.xls'
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, xls.sheet_names[0])

st.title("Mahalle Nüfus Analizi")

# Sidebar for selecting the neighborhood
mahalleler = df['Mahalle'].unique()
selected_mahalle = st.sidebar.selectbox("Mahalle Seçin:", mahalleler)

mahalle_df = df[df['Mahalle'] == selected_mahalle]

# Calculate annual growth rates
mahalle_df['Nüfus Artış Oranı (%)'] = mahalle_df['nufus'].pct_change() * 100

# Layout for population analysis page
page = st.sidebar.selectbox("Görünüm Seçin", ("Nüfus Analizi","Nüfus Dinamikleri Raporu"))

if page == "Nüfus Analizi":
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

elif page == "Nüfus Dinamikleri Raporu":

    st.subheader("Nüfus Artış ve Azalış Nedenleri")

    st.markdown("""
    ### Nüfus Artış Nedenleri:
    - **Ekonomik Fırsatlar**: Bölgedeki iş ve kariyer olanaklarının artması.
    - **Göçler**: Diğer bölgelerden gelen yeni yerleşimciler ve göçmenler.
    - **Doğum Oranları**: Yüksek doğum oranları ve genç nüfus yapısı.
    - **Altyapı İyileştirmeleri**: Sağlık hizmetleri, eğitim ve yaşam kalitesindeki artışlar.
    ### Nüfus Azalış Nedenleri:
    - **Ekonomik Faktörler**: İşsizlik, ekonomik durgunluk ve gelir eşitsizliği.
    - **Göçler**: Bölgeden diğer bölgelere göçlerin artması.
    - **Doğum Oranları**: Doğurganlık oranlarının düşmesi ve yaşlanan nüfus yapısı.
    - **Altyapı Sorunları**: Altyapı eksiklikleri ve yaşam kalitesindeki düşüşler.
    - **Çevresel Faktörler**: Doğal afetler ve çevresel değişimler.
    """)
    st.image('img4.png', caption='2007-2023 Nüfus Değişim Tablosu', use_column_width=True)

    st.markdown(
        """
        
        - Özellikle gelişmekte olan ülkelerde, kırsal kesimden büyük şehirlere yönelik göçlerin artmasıyla birlikte, büyük şehirlerin nüfusu hızla artmaktadır. Bu durum, çoğunlukla iş olanakları, eğitim, sağlık hizmetleri gibi faktörlerden dolayı gerçekleşmektedir. İnsanlar, daha iyi iş imkanları, daha iyi eğitim ve yaşam koşulları gibi sebeplerle büyük şehirlere yönelmektedirler.

        - Büyük şehirlere olan nüfus artışı, şehirlerin altyapı ve hizmetlerini geliştirme ihtiyacını da beraberinde getirir. Bu durum, şehir planlaması, konut politikaları, ulaşım ağları gibi alanlarda büyük çaplı yatırımları gerektirir. Aynı zamanda, bu şehirlerdeki yoğun nüfus, çevresel ve sosyal sorunların da artmasına neden olabilir.

        - Sonuç olarak, nüfus artışının büyük şehirlere yönelmesi, hem şehirlerin hem de ülkenin genel kalkınması için önemli bir dinamiktir. Ancak bu süreç yönetilmezse, olumsuz sonuçlar doğurabilir. Bu nedenle, şehir planlaması ve yönetimi, nüfus artışının sürdürülebilir bir şekilde yönetilmesi açısından kritik bir rol oynamaktadır.
        
        """
    )

    st.image('img2.png', caption='2007-2023 Nüfus Değişim Tablosu', use_column_width=True)