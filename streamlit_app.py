import streamlit as st
import joblib
import pandas as pd

# ─── Конфигурация страницы ─────────────────────────────
st.set_page_config(
    page_title="Wine Quality 🍷",
    page_icon="🍷",
    layout="centered"
)

# ─── Загрузка модели ───────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("wine_model.joblib")

model = load_model()

# ─── Заголовок ─────────────────────────────────────────
st.title("🍷 Оценка качества вина")
st.markdown("Введите параметры вина и получите прогноз качества")

st.divider()

# ─── Ввод данных ───────────────────────────────────────
fixed_acidity = st.number_input("Фиксированная кислотность", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Летучая кислотность", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Лимонная кислота", 0.0, 2.0, 0.0)
residual_sugar = st.number_input("Остаточный сахар", 0.0, 20.0, 1.9)
chlorides = st.number_input("Хлориды", 0.0, 0.2, 0.076)
free_sulfur_dioxide = st.number_input("Свободный диоксид серы", 0, 100, 11)
total_sulfur_dioxide = st.number_input("Общий диоксид серы", 0, 300, 34)
density = st.number_input("Плотность", 0.990, 1.005, 0.997)
pH = st.number_input("pH", 2.5, 4.5, 3.51)
sulphates = st.number_input("Сульфаты", 0.0, 2.0, 0.56)
alcohol = st.number_input("Содержание алкоголя", 8.0, 15.0, 9.4)

st.divider()

# ─── Предсказание ──────────────────────────────────────
if st.button("🔍 Проверить вино", use_container_width=True):

    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
        density, pH, sulphates, alcohol
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
        'density', 'pH', 'sulphates', 'alcohol'
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success("✅ Хорошее вино")
    else:
        st.error("❌ Плохое вино")

    st.metric("🎯 Вероятность качества", f"{probability:.1%}")

    st.progress(probability)

    with st.expander("📋 Введённые данные"):
        st.dataframe(input_data)
