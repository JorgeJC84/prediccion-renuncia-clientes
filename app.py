# app.py

import streamlit as st
import pandas as pd
import joblib

# Título de la app
st.title("🔮 Predicción de Churn de Clientes")
st.write("Sube un archivo CSV con los datos de tus nuevos clientes para predecir la probabilidad de cancelación (churn).")

# Subir archivo CSV
uploaded_file = st.file_uploader("📂 Sube tu archivo CSV", type="csv")

if uploaded_file is not None:
    # Leer archivo
    nuevos = pd.read_csv(uploaded_file)

    # Mostrar datos cargados
    st.subheader("📋 Vista previa de los datos")
    st.dataframe(nuevos)

    # Asegurar columnas exactas (ajusta si tu modelo tiene otras columnas o nombres con doble espacio)
    columnas_modelo = ['Seconds of Use', 'Frequency of use', 'Customer Value',
                       'Frequency of SMS', 'Distinct Called Numbers',
                       'Subscription  Length', 'Status']  # ¡Ojo! hay doble espacio en 'Subscription  Length'

    try:
        nuevos = nuevos[columnas_modelo]

        # Cargar modelo entrenado
        modelo = joblib.load("modelo_churn_rf.pkl")

        # Predecir probabilidad
        nuevos["Prob_Churn"] = modelo.predict_proba(nuevos)[:, 1]

        # Regla de acción
        nuevos["Requiere acción"] = nuevos["Prob_Churn"].apply(
            lambda x: "✅ Enviar promoción" if x > 0.7 else "❌ No urgente"
        )

        # Mostrar resultado
        st.subheader("📊 Resultados de la predicción")
        st.dataframe(nuevos)

        # Descargar resultados
        csv = nuevos.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Descargar resultados", data=csv, file_name="predicciones_churn.csv", mime='text/csv')

    except Exception as e:
        st.error(f"❗ Error: {e}")
        st.warning("Asegúrate de que el archivo tenga exactamente las columnas esperadas.")
