import math
import re
from io import BytesIO
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

st.set_page_config(page_title="Planeación operativa", layout="wide")

# ============================================================
# Catálogos y constantes
# ============================================================
LAT_CANDIDATES = ["latitud", "latitude", "lat", "y"]
LON_CANDIDATES = ["longitud", "longitude", "lon", "lng", "x"]
ID_CANDIDATES = ["id", "id_registro", "clee", "folio", "uuid"]

MUNICIPIOS_SINALOA = [
    "Ahome",
    "Angostura",
    "Badiraguato",
    "Choix",
    "Concordia",
    "Cosalá",
    "Culiacán",
    "El Fuerte",
    "Elota",
    "Escuinapa",
    "Guasave",
    "Mazatlán",
    "Mocorito",
    "Navolato",
    "Rosario",
    "Salvador Alvarado",
    "San Ignacio",
    "Sinaloa",
    "Juan José Ríos",
    "Eldorado",
]

# Coordenadas aproximadas de cabecera municipal / referencia operativa
COORD_MUNICIPIOS = {
    "Ahome": (25.7905, -108.9859),
    "Angostura": (25.3653, -108.1613),
    "Badiraguato": (25.3642, -107.5500),
    "Choix": (26.7092, -108.3222),
    "Concordia": (23.2867, -106.0647),
    "Cosalá": (24.4147, -106.6906),
    "Culiacán": (24.8091, -107.3940),
    "El Fuerte": (26.4169, -108.6183),
    "Elota": (24.0333, -106.8500),
    "Escuinapa": (22.8322, -105.7775),
    "Guasave": (25.5667, -108.4667),
    "Mazatlán": (23.2494, -106.4111),
    "Mocorito": (25.4833, -107.9167),
    "Navolato": (24.7647, -107.6944),
    "Rosario": (22.9924, -105.8572),
    "Salvador Alvarado": (25.4600, -108.0780),
    "San Ignacio": (23.9408, -106.4178),
    "Sinaloa": (25.8333, -108.2167),
    "Juan José Ríos": (25.7500, -108.8333),
    "Eldorado": (24.3228, -107.3719),
}


# ============================================================
# Utilidades base
# ============================================================
@st.cache_data
def load_excel(file_bytes: bytes, filename: str) -> dict[str, pd.DataFrame]:
    xls = pd.ExcelFile(BytesIO(file_bytes), engine="openpyxl")
    sheets = {}
    for name in xls.sheet_names:
        try:
            sheets[name] = pd.read_excel(BytesIO(file_bytes), sheet_name=name, engine="openpyxl")
        except Exception:
            pass
    return sheets


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def find_column(columns, candidates) -> Optional[str]:
    cols_lower = {str(c).strip().lower(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    return None


def infer_geo_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    df = normalize_columns(df)
    lat_col = find_column(df.columns, LAT_CANDIDATES)
    lon_col = find_column(df.columns, LON_CANDIDATES)
    id_col = find_column(df.columns, ID_CANDIDATES)
    return lat_col, lon_col, id_col


def natural_key(texto):
    if pd.isna(texto):
        return ("",)
    return tuple(
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", str(texto))
    )


def ordenar_natural(lista):
    return sorted(lista, key=natural_key)


def distancia_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    x = (lon2 - lon1) * np.cos((lat1 + lat2) / 2)
    y = lat2 - lat1
    return 6371 * np.sqrt(x * x + y * y)


def clean_geo(df: pd.DataFrame, lat_col: str, lon_col: str) -> pd.DataFrame:
    out = df.copy()
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    out["coord_no_nulas"] = out[lat_col].notna() & out[lon_col].notna()
    out["coord_rango_general"] = (
        out[lon_col].between(-118, -100, inclusive="both")
        & out[lat_col].between(18, 33, inclusive="both")
    )
    out["coord_sinaloa_aprox"] = (
        out[lon_col].between(-109.90, -105.20, inclusive="both")
        & out[lat_col].between(22.20, 27.40, inclusive="both")
    )
    out["coord_valida"] = (
        out["coord_no_nulas"]
        & out["coord_rango_general"]
        & out["coord_sinaloa_aprox"]
    )
    return out


def construir_plantilla(num_entrevistadores: int, num_supervisores: int) -> pd.DataFrame:
    entrevistadores = [f"E{i+1}" for i in range(num_entrevistadores)]
    supervisores = [f"S{i+1}" for i in range(num_supervisores)]
    bloques = np.array_split(np.array(entrevistadores, dtype=object), num_supervisores)

    filas = []
    for i, bloque in enumerate(bloques):
        sup = supervisores[i]
        roe = f"R{(i // 3) + 1}"
        for ent in bloque:
            filas.append(
                {
                    "ENTREVISTADOR": str(ent),
                    "SUPERV": sup,
                    "ROE": roe,
                    "MUNICIPIO_RADICACION": "Culiacán",
                }
            )
    return pd.DataFrame(filas)


def generar_alertas(
    df: pd.DataFrame,
    radio_max_radicacion: float = 25.0,
) -> pd.DataFrame:
    df = df.copy()

    if "distancia_centroide" not in df.columns:
        df["distancia_centroide"] = np.nan
    if "distancia_radicacion_km" not in df.columns:
        df["distancia_radicacion_km"] = np.nan

    df["alerta"] = "OK"
    df.loc[~df["coord_no_nulas"], "alerta"] = "COORDENADAS VACIAS"
    df.loc[df["coord_no_nulas"] & ~df["coord_rango_general"], "alerta"] = "COORDENADAS FUERA DE RANGO"
    df.loc[
        df["coord_no_nulas"] & df["coord_rango_general"] & ~df["coord_sinaloa_aprox"],
        "alerta"
    ] = "REVISAR UBICACION"

    validas = df["coord_valida"] & df["distancia_centroide"].notna()
    if validas.any():
        umbral = df.loc[validas, "distancia_centroide"].quantile(0.98)
        df.loc[
            validas & (df["distancia_centroide"] > umbral),
            "alerta"
        ] = "REVISAR DISPERSION"

    mask_rad = (
        df["distancia_radicacion_km"].notna()
        & (df["distancia_radicacion_km"] > radio_max_radicacion)
    )
    df.loc[mask_rad & (df["alerta"] == "OK"), "alerta"] = "LEJOS DE RADICACION"
    return df


def asignacion_balanceada_geografica(
    df_validos: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    peso_radicacion: float = 2.0,
) -> pd.DataFrame:
    """
    Asignación compacta con capacidad:
    - inicia con centroides geográficos
    - incorpora radicación del entrevistador
    - asigna respetando cupos balanceados
    - itera para compactar
    """
    df_validos = df_validos.copy().reset_index(drop=True)
    plantilla = plantilla.copy()

    entrevistadores = ordenar_natural(plantilla["ENTREVISTADOR"].tolist())
    plantilla = plantilla.set_index("ENTREVISTADOR").loc[entrevistadores].reset_index()

    n = len(df_validos)
    k = len(entrevistadores)
    coords = df_validos[[lat_col, lon_col]].to_numpy()

    modelo = KMeans(n_clusters=k, random_state=42, n_init=15)
    _ = modelo.fit_predict(coords)
    centroides = modelo.cluster_centers_.copy()

    base_cupo = n // k
    residuo = n % k
    cupos = {
        entrevistadores[i]: base_cupo + (1 if i < residuo else 0)
        for i in range(k)
    }

    rad_coords = plantilla[["LAT_RADICACION", "LON_RADICACION"]].to_numpy()
    asignados = None

    for _ in range(8):
        costo_cluster = np.sqrt(
            ((coords[:, None, :] - centroides[None, :, :]) ** 2).sum(axis=2)
        )
        costo_radicacion = np.zeros((n, k))

        for j in range(k):
            costo_radicacion[:, j] = distancia_km(
                coords[:, 0],
                coords[:, 1],
                rad_coords[j, 0],
                rad_coords[j, 1],
            )

        costo_total = costo_cluster + (peso_radicacion * costo_radicacion / 100.0)
        preferencias = np.argsort(costo_total, axis=1)

        nuevos_asignados = [None] * n
        ocupacion = {ent: 0 for ent in entrevistadores}
        prioridad_puntos = np.argsort(np.min(costo_total, axis=1))

        for idx in prioridad_puntos:
            for pref in preferencias[idx]:
                ent = entrevistadores[pref]
                if ocupacion[ent] < cupos[ent]:
                    nuevos_asignados[idx] = ent
                    ocupacion[ent] += 1
                    break

        for i in range(n):
            if nuevos_asignados[i] is None:
                for ent in entrevistadores:
                    if ocupacion[ent] < cupos[ent]:
                        nuevos_asignados[i] = ent
                        ocupacion[ent] += 1
                        break

        asignados = nuevos_asignados

        for j, ent in enumerate(entrevistadores):
            mask = np.array(asignados) == ent
            if mask.any():
                centroides[j] = coords[mask].mean(axis=0)

    df_validos["asignado_a"] = asignados

    mapa_sup = dict(zip(plantilla["ENTREVISTADOR"], plantilla["SUPERV"]))
    mapa_roe = dict(zip(plantilla["ENTREVISTADOR"], plantilla["ROE"]))
    mapa_mun = dict(zip(plantilla["ENTREVISTADOR"], plantilla["MUNICIPIO_RADICACION"]))
    mapa_lat_rad = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LAT_RADICACION"]))
    mapa_lon_rad = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LON_RADICACION"]))

    df_validos["SUPERV"] = df_validos["asignado_a"].map(mapa_sup)
    df_validos["ROE"] = df_validos["asignado_a"].map(mapa_roe)
    df_validos["MUNICIPIO_RADICACION"] = df_validos["asignado_a"].map(mapa_mun)
    df_validos["LAT_RADICACION"] = df_validos["asignado_a"].map(mapa_lat_rad)
    df_validos["LON_RADICACION"] = df_validos["asignado_a"].map(mapa_lon_rad)

    centroides_finales = {}
    for ent in entrevistadores:
        subset = df_validos[df_validos["asignado_a"] == ent]
        centroides_finales[ent] = (
            subset[lat_col].mean(),
            subset[lon_col].mean(),
        )

    df_validos["distancia_centroide"] = df_validos.apply(
        lambda row: np.sqrt(
            (row[lat_col] - centroides_finales[row["asignado_a"]][0]) ** 2
            + (row[lon_col] - centroides_finales[row["asignado_a"]][1]) ** 2
        ),
        axis=1,
    )

    df_validos["distancia_radicacion_km"] = df_validos.apply(
        lambda row: distancia_km(
            row[lat_col],
            row[lon_col],
            row["LAT_RADICACION"],
            row["LON_RADICACION"],
        ),
        axis=1,
    )

    df_validos["estatus_asignacion"] = "Asignado"
    return df_validos


def build_summary(
    df_asig: pd.DataFrame,
    min_per: int,
    max_per: int,
    plantilla: pd.DataFrame,
) -> pd.DataFrame:
    resumen = (
        df_asig.groupby(
            ["asignado_a", "SUPERV", "ROE", "MUNICIPIO_RADICACION"],
            dropna=False
        )
        .size()
        .reset_index(name="registros")
        .rename(columns={"asignado_a": "entrevistador"})
    )

    base = plantilla[
        ["ENTREVISTADOR", "SUPERV", "ROE", "MUNICIPIO_RADICACION"]
    ].rename(columns={"ENTREVISTADOR": "entrevistador"})

    resumen = base.merge(
        resumen,
        on=["entrevistador", "SUPERV", "ROE", "MUNICIPIO_RADICACION"],
        how="left",
    )
    resumen["registros"] = resumen["registros"].fillna(0).astype(int)
    resumen["minimo"] = min_per
    resumen["maximo"] = max_per
    resumen["capacidad_restante"] = max_per - resumen["registros"]
    resumen["estado"] = np.select(
        [resumen["registros"] < min_per, resumen["registros"] > max_per],
        ["Debajo del mínimo", "Encima del máximo"],
        default="OK",
    )
    resumen["orden_ent"] = resumen["entrevistador"].map(natural_key)
    return resumen.sort_values("orden_ent").drop(columns="orden_ent").reset_index(drop=True)


def dataframe_to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return output.read()


def construir_mapa(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: str,
    id_col: Optional[str],
):
    category_orders = {}
    if color_col == "asignado_a":
        category_orders[color_col] = ordenar_natural(
            [x for x in df[color_col].dropna().unique().tolist()]
        )

    fig = px.scatter_map(
        df,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        category_orders=category_orders,
        hover_name=id_col if id_col in df.columns else None,
        hover_data={
            "asignado_a": True if "asignado_a" in df.columns else False,
            "SUPERV": True if "SUPERV" in df.columns else False,
            "ROE": True if "ROE" in df.columns else False,
            "MUNICIPIO_RADICACION": True if "MUNICIPIO_RADICACION" in df.columns else False,
            "distancia_radicacion_km": True if "distancia_radicacion_km" in df.columns else False,
            "alerta": True if "alerta" in df.columns else False,
        },
        zoom=6,
        height=700,
        title="Puntos asignados en Sinaloa",
    )

    if "asignado_a" in df.columns:
        resumen_mapa = (
            df[df["asignado_a"].notna()]
            .groupby("asignado_a")
            .agg(
                registros=(lat_col, "count"),
                latitud=(lat_col, "mean"),
                longitud=(lon_col, "mean"),
            )
            .reset_index()
        )
        resumen_mapa["orden_ent"] = resumen_mapa["asignado_a"].map(natural_key)
        resumen_mapa = resumen_mapa.sort_values("orden_ent")

        fig.add_trace(
            go.Scattermap(
                lat=resumen_mapa["latitud"],
                lon=resumen_mapa["longitud"],
                mode="text",
                text=[f"{r.asignado_a}: {int(r.registros)}" for r in resumen_mapa.itertuples()],
                textfont={"size": 13},
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(map_style="carto-positron")
    fig.update_layout(margin={"r": 0, "t": 50, "l": 0, "b": 0})
    return fig


# ============================================================
# Interfaz
# ============================================================
st.title("Planeación operativa de entrevistadores")
st.caption(
    "Versión con radicación municipal, asignación compacta por cercanía y penalización por dispersión."
)

uploaded = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info(
        "La app detecta hojas y columnas como X/Y, latitud/longitud o CLEE, y permite construir la planeación "
        "con base en radicaciones de entrevistadores en municipios de Sinaloa."
    )
    st.stop()

file_bytes = uploaded.getvalue()
sheets = load_excel(file_bytes, uploaded.name)

if not sheets:
    st.error("No fue posible leer hojas del archivo.")
    st.stop()

st.sidebar.header("Configuración")
selected_sheet = st.sidebar.selectbox("Hoja de trabajo", list(sheets.keys()), index=0)
df_raw = normalize_columns(sheets[selected_sheet])

lat_guess, lon_guess, id_guess = infer_geo_columns(df_raw)
lat_options = [None] + list(df_raw.columns)
lon_options = [None] + list(df_raw.columns)
id_options = [None] + list(df_raw.columns)

lat_col = st.sidebar.selectbox(
    "Columna de latitud",
    lat_options,
    index=lat_options.index(lat_guess) if lat_guess in df_raw.columns else 0,
)
lon_col = st.sidebar.selectbox(
    "Columna de longitud",
    lon_options,
    index=lon_options.index(lon_guess) if lon_guess in df_raw.columns else 0,
)
id_col = st.sidebar.selectbox(
    "Columna identificador",
    id_options,
    index=id_options.index(id_guess) if id_guess in df_raw.columns else 0,
)

priority_candidates = [None] + list(df_raw.columns)
default_priority = "PRIORIDAD" if "PRIORIDAD" in df_raw.columns else None
priority_col = st.sidebar.selectbox(
    "Columna de prioridad (opcional)",
    priority_candidates,
    index=priority_candidates.index(default_priority) if default_priority in priority_candidates else 0,
)

st.sidebar.subheader("Parámetros operativos")
num_interviewers = st.sidebar.number_input("Número de entrevistadores", min_value=1, value=6, step=1)
min_per = st.sidebar.number_input("Mínimo por entrevistador", min_value=0, value=40, step=1)
max_per = st.sidebar.number_input("Máximo por entrevistador", min_value=1, value=60, step=1)
num_supervisores = st.sidebar.number_input("Supervisores", min_value=1, value=3, step=1)
peso_radicacion = st.sidebar.number_input(
    "Peso de radicación vs dispersión",
    min_value=0.0,
    value=2.5,
    step=0.5,
    help="Mayor valor = más castigo a asignar puntos lejos de la radicación.",
)
radio_max_radicacion = st.sidebar.number_input(
    "Radio máximo de referencia desde radicación (km)",
    min_value=1.0,
    value=30.0,
    step=1.0,
)

with st.expander("Plantilla operativa", expanded=True):
    plantilla_default = construir_plantilla(
        int(num_interviewers),
        int(num_supervisores),
    )
    plantilla_edit = st.data_editor(
        plantilla_default,
        column_config={
            "MUNICIPIO_RADICACION": st.column_config.SelectboxColumn(
                "Municipio de radicación",
                options=MUNICIPIOS_SINALOA,
                required=True,
            )
        },
        num_rows="fixed",
        use_container_width=True,
        key="editor_plantilla",
    )
    plantilla_edit["LAT_RADICACION"] = plantilla_edit["MUNICIPIO_RADICACION"].map(
        lambda x: COORD_MUNICIPIOS[x][0]
    )
    plantilla_edit["LON_RADICACION"] = plantilla_edit["MUNICIPIO_RADICACION"].map(
        lambda x: COORD_MUNICIPIOS[x][1]
    )
    interviewer_names = plantilla_edit["ENTREVISTADOR"].tolist()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Hojas detectadas", len(sheets))
with col2:
    st.metric("Registros en hoja", len(df_raw))
with col3:
    st.metric("Columnas", len(df_raw.columns))
with col4:
    st.metric("Entrevistadores", len(interviewer_names))

st.subheader("Vista previa del archivo")
st.dataframe(df_raw.head(20), use_container_width=True)

if not lat_col or not lon_col:
    st.error("Selecciona las columnas de latitud y longitud para continuar.")
    st.stop()

df_geo = clean_geo(df_raw, lat_col, lon_col)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Coordenadas válidas", int(df_geo["coord_valida"].sum()))
with c2:
    st.metric("Coordenadas a revisar", int((~df_geo["coord_valida"]).sum()))
with c3:
    st.metric("Sin coordenadas", int(df_geo[[lat_col, lon_col]].isna().any(axis=1).sum()))

run = st.button("Ejecutar asignación compacta", type="primary")

if run:
    try:
        df_work = df_geo.copy()
        df_work["row_id"] = np.arange(len(df_work))

        validos = df_work[df_work["coord_valida"]].copy()
        invalidos = df_work[~df_work["coord_valida"]].copy()

        if len(validos) < len(plantilla_edit):
            st.error("No hay suficientes registros válidos para generar un grupo compacto por entrevistador.")
            st.stop()

        capacidad_total = len(interviewer_names) * int(max_per)
        if len(validos) > capacidad_total:
            st.warning(
                f"Hay {len(validos):,} registros válidos y la capacidad máxima es {capacidad_total:,}. "
                "En esta versión se reparte de forma balanceada entre todos los entrevistadores disponibles."
            )

        asignados = asignacion_balanceada_geografica(
            validos,
            plantilla_edit,
            lat_col=lat_col,
            lon_col=lon_col,
            peso_radicacion=float(peso_radicacion),
        )

        asignados = generar_alertas(
            asignados,
            radio_max_radicacion=float(radio_max_radicacion),
        )

        if len(invalidos) > 0:
            invalidos["asignado_a"] = None
            invalidos["estatus_asignacion"] = "Coordenada inválida o a revisar"
            invalidos["SUPERV"] = None
            invalidos["ROE"] = None
            invalidos["MUNICIPIO_RADICACION"] = None
            invalidos["LAT_RADICACION"] = np.nan
            invalidos["LON_RADICACION"] = np.nan
            invalidos["distancia_centroide"] = np.nan
            invalidos["distancia_radicacion_km"] = np.nan
            invalidos = generar_alertas(
                invalidos,
                radio_max_radicacion=float(radio_max_radicacion),
            )
            resultado = pd.concat([asignados, invalidos], ignore_index=True)
        else:
            resultado = asignados.copy()

        resumen = build_summary(
            resultado[resultado["estatus_asignacion"] == "Asignado"].copy(),
            min_per=int(min_per),
            max_per=int(max_per),
            plantilla=plantilla_edit,
        )

        st.success("Asignación compacta generada.")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.metric("Asignados", int((resultado["estatus_asignacion"] == "Asignado").sum()))
        with k2:
            st.metric("Sin asignar", int(resultado["asignado_a"].isna().sum()))
        with k3:
            st.metric("Promedio por entrevistador", round(resumen["registros"].mean(), 1) if len(resumen) else 0)
        with k4:
            st.metric("Máximo observado", int(resumen["registros"].max()) if len(resumen) else 0)

        st.subheader("Resumen por entrevistador")
        st.dataframe(resumen, use_container_width=True)

        st.subheader("Validación visual de cargas")
        fig_bar = px.bar(
            resumen,
            x="entrevistador",
            y="registros",
            color="estado",
            text="registros",
            category_orders={"entrevistador": ordenar_natural(resumen["entrevistador"].tolist())},
            title="Registros asignados por entrevistador",
        )
        fig_bar.add_hline(y=min_per, line_dash="dash", annotation_text="mínimo")
        fig_bar.add_hline(y=max_per, line_dash="dash", annotation_text="máximo")
        st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Mapa operativo de puntos")
        map_df = resultado.copy()
        map_df["grupo_mapa"] = map_df["asignado_a"].fillna(map_df["alerta"]).fillna("SIN DATOS")
        fig_map = construir_mapa(
            map_df[map_df[lat_col].notna() & map_df[lon_col].notna()].copy(),
            lat_col,
            lon_col,
            "grupo_mapa",
            id_col,
        )
        st.plotly_chart(fig_map, use_container_width=True)

        st.subheader("Inconsistencias y alertas")
        inconsistencias = (
            resultado[resultado["alerta"] != "OK"].copy()
            if "alerta" in resultado.columns
            else pd.DataFrame()
        )
        st.dataframe(inconsistencias, use_container_width=True)

        st.subheader("Detalle de registros")
        st.dataframe(resultado, use_container_width=True)

        export_bytes = dataframe_to_excel_bytes(
            {
                "asignacion": resultado,
                "resumen_entrevistador": resumen,
                "plantilla": plantilla_edit,
                "parametros": pd.DataFrame(
                    {
                        "parametro": [
                            "hoja_seleccionada",
                            "lat_col",
                            "lon_col",
                            "id_col",
                            "priority_col",
                            "entrevistadores",
                            "min_por_entrevistador",
                            "max_por_entrevistador",
                            "supervisores",
                            "peso_radicacion",
                            "radio_max_radicacion_km",
                        ],
                        "valor": [
                            selected_sheet,
                            lat_col,
                            lon_col,
                            id_col,
                            priority_col,
                            len(interviewer_names),
                            min_per,
                            max_per,
                            num_supervisores,
                            peso_radicacion,
                            radio_max_radicacion,
                        ],
                    }
                ),
            }
        )

        st.download_button(
            "Descargar resultado en Excel",
            data=export_bytes,
            file_name="planeacion_operativa_resultado.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    except Exception as e:
        st.exception(e)

st.markdown("---")
st.markdown(
    "**Incluye:** radicación por municipio, penalización por lejanía a la base del entrevistador "
    "y asignación compacta por cercanía."
)
