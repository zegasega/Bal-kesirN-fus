import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import folium
from streamlit_folium import st_folium
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load the Excel file
file_path = 'merkez.xls'
xls = pd.ExcelFile(file_path)
df = pd.read_excel(xls, xls.sheet_names[0])
text_df = pd.read_csv('text_data.csv')
# Stopword listesini indirme
nltk.download('stopwords')
turkish_stopwords = stopwords.words('turkish')

# Sidebar for selecting the neighborhood
mahalleler = df['Mahalle'].unique()
selected_mahalle = st.sidebar.selectbox("Mahalle Seçin:", mahalleler)

mahalle_df = df[df['Mahalle'] == selected_mahalle]

# Calculate annual growth rates
mahalle_df['Nüfus Artış Oranı (%)'] = mahalle_df['nufus'].pct_change() * 100

# Layout for population analysis page
page = st.sidebar.selectbox("Sayfa Seçin", ("Nüfus Analizi", "Nüfus Dinamikleri Raporu", "Balıkesir Haritası","Text Classification"))

if page == "Text Classification":

        st.title("Metin Sınıflandırma Uygulaması")




        # Stopword'leri temizleme fonksiyonu
        def remove_stopwords(text):
            if pd.isna(text):
                return ''
            tokens = text.split()
            tokens = [word for word in tokens if word.lower() not in turkish_stopwords]
            tokens = [word for word in tokens if word not in string.punctuation]
            return ' '.join(tokens)


        # Veriyi temizleme
        text_df['cleaned_text'] = text_df['text'].apply(remove_stopwords)

        # Gerekli sütunları seçme
        text_df = text_df[['category', 'cleaned_text']]
        text_df = text_df.rename(columns={'category': 'label', 'cleaned_text': 'text'})

        # Eğitim ve test veri setlerini bölme
        train_df, test_df = train_test_split(text_df, test_size=0.2, random_state=42)

        # Metin vektörizasyonu için CountVectorizer kullanma
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label']

        X_test = vectorizer.transform(test_df['text'])
        y_test = test_df['label']

        # Naive Bayes modelini eğitme
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, y_train)

        # Kullanıcıdan metin girişi alma
        user_input = st.text_area("Bir metin girin:", "",height=200)
        if user_input:
            # Girilen metni vektörize etme
            user_input_vectorized = vectorizer.transform([user_input])

            # Tahmin yapma
            predicted_label = nb_classifier.predict(user_input_vectorized)[0]

            st.write(f"Tahmin edilen etiket: {predicted_label}")

        # Model performansını gösterme
        if st.checkbox("Model Performansını Göster"):
            y_pred = nb_classifier.predict(X_test)
            st.text("Sınıflandırma Raporu:")
            st.text(classification_report(y_test, y_pred))


if page == "Balıkesir Haritası":
    st.title("Balıkesir Haritası")

    # Center the map on Balıkesir
    map_center = [39.6484, 27.8826]  # Coordinates for Balıkesir, Turkey
    m = folium.Map(location=map_center, zoom_start=14)

    # Add a marker for Balıkesir city center
    folium.Marker(
        location=map_center,
        popup="Balıkesir",
        icon=folium.Icon(icon="info-sign")
    ).add_to(m)

    # Render the map in Streamlit
    st_folium(m, width=1600, height=800)

if page == "Nüfus Analizi":
    st.title("Merkez Mahalle Nüfus Analizi")
    st.subheader(f"{selected_mahalle} Mahallesinin Nüfus Grafiği ve Analizi")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Nüfus Grafiği (2007-2023)")
        fig1, ax1 = plt.subplots()
        ax1.plot(mahalle_df['yil'], mahalle_df['nufus'], marker='o', label='Gerçek Nüfus')
        ax1.set_xlabel("Yıl")
        ax1.set_ylabel("Nüfus")
        st.pyplot(fig1)

    with col2:
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

    with col3:
        st.subheader("Yıllara Göre Nüfus Artış Oranı Grafiği")
        fig3, ax3 = plt.subplots()
        ax3.plot(mahalle_df['yil'], mahalle_df['Nüfus Artış Oranı (%)'], marker='o', linestyle='-', color='b')
        ax3.set_xlabel("Yıl")
        ax3.set_ylabel("Nüfus Artış Oranı (%)")
        st.pyplot(fig3)

    st.subheader("Yıllara Göre Nüfus Artış Oranı (%)")
    st.table(mahalle_df[['yil', 'Nüfus Artış Oranı (%)']])

elif page == "Nüfus Dinamikleri Raporu":
    st.subheader("Nüfus Artış ve Azalış Nedenleri")

    st.markdown("""
    ### Nüfus Artış Nedenleri:
    - **Ekonomik Fırsatlar**: Bölgedeki iş ve kariyer olanaklarının artması, daha fazla insanı bölgeye çekmektedir.
    - **Göçler**: Diğer bölgelerden gelen yeni yerleşimciler ve göçmenler, nüfusu artıran önemli bir faktördür.
    - **Doğum Oranları**: Yüksek doğum oranları ve genç nüfus yapısı, nüfusun artmasına katkı sağlar.
    - **Altyapı İyileştirmeleri**: Sağlık hizmetleri, eğitim ve genel yaşam kalitesindeki iyileşmeler, daha fazla insanın bölgede kalmasını sağlar.

    ### Nüfus Azalış Nedenleri:
    - **Ekonomik Faktörler**: İşsizlik, ekonomik durgunluk ve gelir eşitsizliği, insanların bölgeden göç etmesine neden olabilir.
    - **Göçler**: Bölgede yaşayan insanların, daha iyi fırsatlar için başka bölgelere göç etmeleri nüfus azalmasına neden olabilir.
    - **Doğum Oranları**: Doğurganlık oranlarının düşmesi ve yaşlanan nüfus yapısı, nüfusun azalmasına yol açar.
    - **Altyapı Sorunları**: Altyapı eksiklikleri ve yaşam kalitesindeki düşüşler, insanların bölgeden taşınmasına neden olabilir.
    - **Çevresel Faktörler**: Doğal afetler ve çevresel değişimler, bölgenin yaşanabilirliğini azaltarak nüfus azalmasına neden olabilir.
    """)

    st.image('img4.png', caption='2007-2023 Nüfus Değişim Tablosu', width=800)

    st.markdown("""
    Nüfus artışının büyük şehirlere yönelmesi, özellikle gelişmekte olan ülkelerde yaygın bir olgudur.

    Kırsal kesimden büyük şehirlere olan göçlerin artması, büyük şehirlerin nüfusunun hızla artmasına neden olur.
    Bu durum, iş olanakları, eğitim ve sağlık hizmetleri gibi faktörlerin etkisiyle gerçekleşir.

    İnsanlar, daha iyi iş imkanları, daha iyi eğitim ve yaşam koşulları gibi sebeplerle büyük şehirlere yönelmektedirler.

    Büyük şehirlere olan nüfus artışı, şehirlerin altyapı ve hizmetlerini geliştirme ihtiyacını da beraberinde getirir.
    Bu durum, şehir planlaması, konut politikaları, ulaşım ağları gibi alanlarda büyük çaplı yatırımları gerektirir.
    Aynı zamanda, bu şehirlerdeki yoğun nüfus, çevresel ve sosyal sorunların da artmasına neden olabilir.

    Sonuç olarak, nüfus artışının büyük şehirlere yönelmesi, hem şehirlerin hem de ülkenin genel kalkınması için önemli bir dinamiktir. Ancak bu süreç doğru şekilde yönetilmezse, olumsuz sonuçlar doğurabilir. Bu nedenle, şehir planlaması ve yönetimi, nüfus artışının sürdürülebilir bir şekilde yönetilmesi açısından kritik bir rol oynamaktadır.

    ### Büyük Şehirlerde Nüfus Artışının Etkileri:
    - **İş Olanakları**: Daha fazla insan iş gücüne katılmakta ve ekonomik büyümeye katkıda bulunmaktadır.
    - **Eğitim ve Sağlık**: Daha iyi eğitim ve sağlık hizmetlerine erişim, genel yaşam kalitesini artırmaktadır.
    - **Altyapı Gelişimi**: Artan nüfus, altyapı ve ulaşım sistemlerinin genişletilmesi ve iyileştirilmesini gerektirir.
    - **Çevresel ve Sosyal Sorunlar**: Yoğun nüfus, trafik sıkışıklığı, hava kirliliği ve sosyal hizmetlerde yetersizlik gibi sorunlara yol açabilir.

    Bu bağlamda, şehirlerin planlı ve sürdürülebilir bir şekilde büyümesi, hem mevcut hem de gelecekteki nesiller için daha iyi yaşam koşulları sağlar.
    """)

    st.image('img2.png', caption='2007-2023 Nüfus Değişim Tablosu', width=800)
