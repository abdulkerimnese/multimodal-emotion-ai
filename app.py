import streamlit as st
import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
from PIL import Image
import mediapipe as mp

# --- SAYFA VE UI AYARLARI ---
st.set_page_config(page_title="Multimodal Fusion AI", page_icon="🧠", layout="wide")

# CSS Enjeksiyonu (Streamlit logolarını gizleme ve Buton tasarımı)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tüm Butonların Tasarımı */
    .stButton > button {
        background-color: #EF553B !important;
        color: #FFFFFF !important; /* Yazı rengini kesin beyaz yapar */
        font-weight: bold !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 15px !important;
        transition: all 0.3s ease !important;
    }
    
    /* Buton içindeki yazıları ve ikonları zorla beyaz yap */
    .stButton > button * {
        color: #FFFFFF !important;
    }
    
    /* Fareyle üzerine gelindiğinde (Hover) */
    .stButton > button:hover {
        background-color: #D4442E !important;
        box-shadow: 0px 4px 10px rgba(239, 85, 59, 0.4) !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- POPUP (BİLGİLENDİRME MODALI) ---
@st.dialog("👋 Fusion AI Sistemine Hoş Geldiniz!")
def welcome_dialog():
    st.markdown("""
    Bu yapay zeka sistemi, insanların anlık duygularını anlamak için **hem yüz ifadelerini (CNN)** hem de **yazdıkları metinleri (NLP)** eşzamanlı olarak analiz eder.
    
    🎯 **Nasıl Kullanılır?**
    1. **Görüntü:** Sol taraftan kameranızı açıp bir mimik yapın (veya fotoğraf yükleyin).
    2. **Metin:** Sağ taraftaki kutuya o anki hislerinizi (veya mimiğinize tamamen zıt bir cümle) yazın.
    3. **Analiz:** Aşağıdaki kırmızı butona tıklayın.
    
    *Sistem, Mediapipe teknolojisi ile yüzünüzü saniyeler içinde tarayıp arka planı silecek ve iki farklı yapay zeka modelini harmanlayarak nihai duygunuzu tahmin edecektir.*
    """)
    if st.button("Anladım, Kapat 🚀", use_container_width=True):
        st.session_state.welcome_shown = True
        st.rerun()

# Site açıldığında Popup'ı göster (Sadece 1 kere çalışır)
if "welcome_shown" not in st.session_state:
    welcome_dialog()


# --- MEDIAPIPE VE MODELLER ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=1)
contour_drawing_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=3)

@st.cache_resource
def load_all_models():
    cnn = load_model('cnn_emotion_model.h5')
    nlp = joblib.load('nlp_model_v2.pkl')
    tfidf = joblib.load('tfidf_vectorizer_v2.pkl')
    return cnn, nlp, tfidf

with st.spinner("Sistem Başlatılıyor... Modeller Yükleniyor..."):
    cnn_model, nlp_model, tfidf_vectorizer = load_all_models()

cnn_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def process_with_mediapipe(image_bytes):
    img_array = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    img_drawn = img_rgb.copy()
    cropped_gray_for_cnn = None
    pip_image = None
    face_found = False
    
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            face_found = True
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image=img_drawn, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec=mesh_drawing_spec, connection_drawing_spec=mesh_drawing_spec)
                mp_drawing.draw_landmarks(image=img_drawn, landmark_list=face_landmarks, connections=mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=contour_drawing_spec)
                
                h, w, _ = img_rgb.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)
                
                pad_x, pad_y = int((x_max - x_min) * 0.2), int((y_max - y_min) * 0.2)
                x_min, y_min = max(0, x_min - pad_x), max(0, y_min - pad_y)
                x_max, y_max = min(w, x_max + pad_x), min(h, y_max + pad_y)
                
                cropped_face_bgr = img_bgr[y_min:y_max, x_min:x_max]
                if cropped_face_bgr.size != 0:
                    cropped_gray_for_cnn = cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2GRAY)
                    pip_image = cv2.resize(cv2.cvtColor(cropped_face_bgr, cv2.COLOR_BGR2RGB), (150, 150))
                else:
                    cropped_gray_for_cnn = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            cropped_gray_for_cnn = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            
    return img_drawn, cropped_gray_for_cnn, pip_image, face_found

def multimodal_predict(text, face_gray_img, text_weight, image_weight):
    text_vectorized = tfidf_vectorizer.transform([text])
    nlp_probs = nlp_model.predict_proba(text_vectorized)[0] 
    nlp_score_negative, nlp_score_positive = nlp_probs[0], nlp_probs[1]
    
    img_resized = cv2.resize(face_gray_img, (48, 48))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 48, 48, 1))
    cnn_probs = cnn_model.predict(img_reshaped, verbose=0)[0]
    
    final_probs = np.zeros(7)
    for i, emotion in enumerate(cnn_classes):
        if emotion in ['Happy', 'Surprise']:
            final_probs[i] = (cnn_probs[i] * image_weight) + (nlp_score_positive * text_weight)
        elif emotion in ['Angry', 'Disgust', 'Fear', 'Sad']:
            final_probs[i] = (cnn_probs[i] * image_weight) + (nlp_score_negative * text_weight)
        else:
            final_probs[i] = cnn_probs[i] 

    return cnn_classes[np.argmax(final_probs)], final_probs

# --- ANA EKRAN TASARIMI ---
st.title("🧠 Multimodal Fusion AI")
st.markdown("Yüz ifadenizi ve metninizi harmanlayarak gerçek duygunuzu analiz eden yapay zeka arayüzü.")
st.divider()

# --- YAN MENÜ (SIDEBAR) ---
st.sidebar.header("⚙️ Model Parametreleri")
st.sidebar.markdown("Karar mekanizmasında hangi modelin daha baskın olacağını seçin:")
image_weight = st.sidebar.slider("📸 Görüntü (CNN) Ağırlığı", 0.0, 1.0, 0.6, 0.1)
text_weight = st.sidebar.slider("📝 Metin (NLP) Ağırlığı", 0.0, 1.0, 0.4, 0.1)

st.sidebar.divider()

# Bilgi Ekranını Tekrar Açma Butonu
st.sidebar.header("ℹ️ Yardım & Bilgi")
if st.sidebar.button("📖 Sistemi Nasıl Kullanırım?", use_container_width=True):
    welcome_dialog()

st.sidebar.divider()
st.sidebar.info("💡 **İpucu:** Zıtlık testi yapmak için mimiklerinize ters düşen metinler yazıp sonuçları gözlemleyin.")

# --- VERİ GİRİŞ ALANI ---
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### 1. Görsel Girdi (Mimik)")
    tab1, tab2 = st.tabs(["📸 Kamera", "📁 Dosya Yükle"])
    with tab1: camera_photo = st.camera_input("Kameraya bakarak bir ifade verin", label_visibility="collapsed")
    with tab2: uploaded_file = st.file_uploader("Veya bilgisayarınızdan seçin", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    final_image = camera_photo if camera_photo is not None else uploaded_file

with col2:
    st.markdown("#### 2. Metin Girdisi (İfade)")
    user_text = st.text_area("Ne düşünüyorsunuz?", height=120, placeholder="Örn: Mülakatımın harika geçeceğine inanıyorum!")
    st.markdown("<br>", unsafe_allow_html=True) # Boşluk
    analyze_button = st.button("🚀 Gelişmiş Analizi Başlat", use_container_width=True)

# --- SONUÇ ALANI ---
if analyze_button:
    if final_image is None or user_text.strip() == "":
        st.error("Lütfen analize başlamadan önce bir fotoğraf sağlayın ve bir metin girin.")
    else:
        st.divider()
        st.markdown("### 📊 Analiz Sonuçları")
        
        with st.spinner("Veriler işleniyor... (Mediapipe & AI Fusion)"):
            image_bytes = final_image.getvalue()
            img_drawn, cropped_gray, pip_image, face_found = process_with_mediapipe(image_bytes)
            
            res_col1, res_col2 = st.columns([1, 1.5], gap="large")
            
            with res_col1:
                if not face_found:
                    st.warning("Yüz algılanamadı! Analiz tüm fotoğraf üzerinden yapıldı.")
                    st.image(img_drawn, use_container_width=True, clamp=True)
                else:
                    st.success("Yüz hatları başarıyla çözümlendi!")
                    st.image(img_drawn, caption="Mediapipe 468-Nokta Yüz Topolojisi", use_container_width=True)
                    st.markdown("**CNN Girdisi (Kırpılmış Orijinal):**")
                    st.image(pip_image, width=120)

            with res_col2:
                predicted_emotion, all_probabilities = multimodal_predict(user_text, cropped_gray, text_weight, image_weight)
                
                st.markdown(f"#### Nihai Yapay Zeka Kararı: <span style='color:#EF553B'>**{predicted_emotion.upper()}**</span>", unsafe_allow_html=True)
                
                colors = ['#636EFA'] * 7
                colors[cnn_classes.index(predicted_emotion)] = '#EF553B' 
                
                fig = go.Figure(data=[go.Bar(
                    x=cnn_classes, 
                    y=all_probabilities, 
                    marker_color=colors,
                    text=np.round(all_probabilities, 3),
                    textposition='outside',
                    textfont=dict(color='white')
                )])
                
                fig.update_layout(
                    xaxis_title="Duygu Sınıfları",
                    yaxis_title="Güven Skoru (Olasılık)",
                    template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white",
                    height=380,
                    margin=dict(l=0, r=0, t=30, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)