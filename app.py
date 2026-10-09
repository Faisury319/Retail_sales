import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler

# Configuración de la página de Streamlit
st.set_page_config(
    page_title="Predicción de Ventas de Retail",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Aplicación de Predicción de Ventas Semanales (SVM)")
st.markdown("""
Esta aplicación web procesa un archivo de datos de ventas en formato **Excel**, aplica la limpieza de datos requerida 
(manejo de outliers, imputación KNN, escalado Min-Max) y predice las ventas semanales utilizando el modelo **SVM Optimizado**.
""")

# Obtener la ruta del directorio actual de ejecución para despliegues independientes (como GitHub)
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
scaler_path = os.path.join(current_dir, 'min_max_scaler.joblib')
model_path = os.path.join(current_dir, 'svm_optimizado.joblib')

# Verificar la existencia de los archivos del modelo y escalador esenciales
if not os.path.exists(scaler_path) or not os.path.exists(model_path):
    st.error("⚠️ No se encontraron los archivos del modelo (`svm_optimizado.joblib`) o del escalador (`min_max_scaler.joblib`) en el directorio de la aplicación.")
    st.info("Asegúrate de subir ambos archivos `.joblib` junto con tu código a tu repositorio de GitHub.")
else:
    # Cargar el escalador y el modelo de manera global
    scaler = joblib.load(scaler_path)
    svm_model = joblib.load(model_path)

    # Widget para que el usuario cargue su archivo de Excel
    uploaded_file = st.file_uploader("Sube tu archivo de datos Excel (.xlsx o .xls)", type=["xlsx", "xls"])

    if uploaded_file is not None:
        try:
            # 1. Leer los datos de Excel
            df_input = pd.read_excel(uploaded_file)
            st.success("¡Archivo cargado correctamente!")
            st.subheader("Vista previa de los datos originales cargados")
            st.dataframe(df_input.head(5))

            # Crear copia para no alterar el DataFrame original expuesto al usuario
            df = df_input.copy()

            with st.spinner("Procesando datos y generando predicciones..."):
                # 2. Convertir variables tipo objeto a categoría
                for col in df.select_dtypes(include=['object']).columns:
                    df[col] = df[col].astype('category')

                # 3. Convertir fecha a datetime y eliminar columnas innecesarias si están presentes
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')
                    df = df.drop(columns=['Date'])
                
                columns_to_drop = ['RecordID', 'FullName', 'Phone', 'ZodiacSign', 'FavoriteColor', 'Hobby']
                df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

                # 4. Calcular 'Store_Total_Sales' si no viene precalculado
                if 'Store_Total_Sales' not in df.columns and 'StoreID' in df.columns and 'Weekly_Sales' in df.columns:
                    df['Store_Total_Sales'] = df.groupby('StoreID')['Weekly_Sales'].transform('sum')
                elif 'Store_Total_Sales' not in df.columns:
                    # Valor por defecto representativo en caso de ausencia
                    df['Store_Total_Sales'] = 1460766.60

                # Guardar 'Weekly_Sales' real temporalmente si existe para evaluar o excluir del procesamiento
                target_col = 'Weekly_Sales'
                has_target = target_col in df.columns
                actual_sales = df[target_col].copy() if has_target else None

                # 5. Tratamiento de Outliers (Reemplazar valores atípicos por Nulos mediante IQR)
                numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
                # Excluir la variable objetivo de la alteración de outliers para predicción directa
                if target_col in numerical_cols:
                    numerical_cols.remove(target_col)

                for col in numerical_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    df.loc[(df[col] < lower_bound) | (df[col] > upper_bound), col] = np.nan

                # 6. Imputación de nulos utilizando KNN Imputer
                # Identificar columnas numéricas con valores faltantes o modificadas por outliers
                cols_to_impute = df[numerical_cols].columns[df[numerical_cols].isnull().any()].tolist()
                if cols_to_impute:
                    imputer = KNNImputer(n_neighbors=5)
                    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

                # Asegurar de que no queden nulos generales en las variables explicativas
                df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

                # 7. Escalado Min-Max
                # Seleccionar las columnas en el mismo orden que espera el scaler
                features_esperadas = [
                    'StoreID', 'DayOfWeek', 'Promotion', 'Holiday', 'Temperature', 
                    'FuelPrice', 'CPI', 'Unemployment', 'Sales_3Months', 'Store_Total_Sales'
                ]

                # Crear columnas faltantes con la mediana histórica si el Excel del usuario no las incluye
                for feat in features_esperadas:
                    if feat not in df.columns:
                        df[feat] = 0.0

                # Reordenar al formato exacto
                df_for_scaling = df[features_esperadas].copy()
                
                # Aplicar el MinMaxScaler precargado
                scaled_array = scaler.transform(df_for_scaling)
                X_processed = pd.DataFrame(scaled_array, columns=features_esperadas)

                # 8. Realizar la predicción con el modelo SVM
                predictions = svm_model.predict(X_processed)

                # Agregar las predicciones al archivo original para descarga
                df_output = df_input.copy()
                df_output['Predicted_Weekly_Sales'] = predictions

            # Mostrar Resultados al usuario
            st.subheader("✨ Resultados de la Predicción")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Total de registros procesados", value=len(df_output))
            with col2:
                st.metric(label="Promedio de Ventas Semanales Predichas", value=f"${predictions.mean():,.2f}")

            st.dataframe(df_output[['RecordID', 'StoreID', 'Date', 'Predicted_Weekly_Sales'] if 'RecordID' in df_output.columns and 'Date' in df_output.columns else df_output.head(20)])

            # Permite al usuario descargar los resultados formateados en un nuevo Excel
            @st.cache_data
            def convert_df_to_excel(df_to_download):
                # Escribir a memoria para evitar dependencias locales de escritura física
                import io
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_to_download.to_excel(writer, index=False, sheet_name='Predicciones')
                return output.getvalue()

            excel_data = convert_df_to_excel(df_output)
            st.download_button(
                label="📥 Descargar archivo con predicciones en Excel",
                data=excel_data,
                file_name="predicciones_ventas_semanales.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        except Exception as e:
            st.error(f"❌ Ocurrió un error al procesar el archivo Excel: {e}")
