# 🔮 Predicción de Renuncia de Clientes

Este proyecto predice si un cliente abandonará el servicio, usando machine learning y una app interactiva en Streamlit.

## 🧠 Modelo
Se entrenó un modelo Random Forest con datos balanceados usando SMOTE.

## 📁 Archivos
- `app.py`: app Streamlit para cargar datos y predecir.
- `modelo_churn_rf.pkl`: modelo entrenado.
- `nuevos_clientes.csv`: ejemplo de datos nuevos.
- `Predicción de Renuncia de Clientes_Jorge_Jeria_G93_2.ipynb`: notebook de análisis y entrenamiento.
- `requirements.txt`: librerías necesarias.

## ▶️ Cómo ejecutar

```bash
pip install -r requirements.txt
streamlit run app.py
```

Sube un archivo CSV y obtendrás la probabilidad de renuncia de clientes.
