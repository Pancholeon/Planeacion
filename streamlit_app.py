import streamlit as st
import pandas as pd

st.set_page_config(page_title="Planeación Operativa", layout="wide")

st.title("Sistema de Planeación Operativa")

st.write("Sube el archivo Excel para comenzar la planeación.")

# Subir archivo
archivo = st.file_uploader("Subir archivo Excel", type=["xlsx", "xlsm", "xls"])

if archivo is not None:

    # Leer todas las hojas
    excel = pd.ExcelFile(archivo)

    st.subheader("Hojas disponibles en el archivo")
    st.write(excel.sheet_names)

    # Seleccionar hoja
    hoja = st.selectbox("Selecciona la hoja que contiene los registros", excel.sheet_names)

    # Leer datos
    df = pd.read_excel(archivo, sheet_name=hoja)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head(20))

    st.write("Número de registros:", df.shape[0])
    st.write("Número de columnas:", df.shape[1])
