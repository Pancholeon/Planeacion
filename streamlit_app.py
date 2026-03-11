import re
from io import BytesIO
from typing import Optional, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.features import DivIcon
from folium.plugins import Draw, Fullscreen
from sklearn.cluster import KMeans
from streamlit_folium import st_folium

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

REGION_MUNICIPIO = {
    "Ahome": "NORTE",
    "El Fuerte": "NORTE",
    "Choix": "NORTE",
    "Juan José Ríos": "NORTE",
    "Guasave": "NORTE-CENTRO",
    "Sinaloa": "NORTE-CENTRO",
    "Angostura": "CENTRO-NORTE",
    "Salvador Alvarado": "CENTRO-NORTE",
    "Mocorito": "CENTRO-NORTE",
    "Badiraguato": "CENTRO",
    "Culiacán": "CENTRO",
    "Navolato": "CENTRO",
    "Eldorado": "CENTRO",
    "Cosalá": "CENTRO-SUR",
    "Elota": "CENTRO-SUR",
    "San Ignacio": "SUR",
    "Mazatlán": "SUR",
    "Concordia": "SUR",
    "Rosario": "SUR",
    "Escuinapa": "SUR",
}

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#393b79", "#637939",
    "#8c6d31", "#843c39", "#7b4173", "#3182bd", "#31a354", "#756bb1",
]

# ============================================================
# Utilidades
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
    return tuple(int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(texto)))


def ordenar_natural(lista):
    return sorted(lista, key=natural_key)


def color_for_entity(name: str) -> str:
    if pd.isna(name):
        return "#808080"
    idx = sum(ord(c) for c in str(name)) % len(PALETTE)
    return PALETTE[idx]


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
    out["coord_valida"] = out["coord_no_nulas"] & out["coord_rango_general"] & out["coord_sinaloa_aprox"]
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


def dataframe_to_excel_bytes(dfs: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
    output.seek(0)
    return output.read()


def prioridad_serie(df: pd.DataFrame, priority_col: Optional[str]) -> pd.Series:
    if priority_col is None or priority_col not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index)

    s = df[priority_col]
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().any():
        fill_value = float(num.min()) if num.notna().any() else 0.0
        return num.fillna(fill_value)

    s2 = s.astype(str).str.strip().str.upper()
    ranking = {
        "MUY ALTA": 5,
        "ALTA": 4,
        "MEDIA": 3,
        "BAJA": 2,
        "MUY BAJA": 1,
    }
    return s2.map(ranking).fillna(0)


def municipio_mas_cercano(lat: float, lon: float) -> Optional[str]:
    mejor = None
    mejor_d = None
    for mun, (mlat, mlon) in COORD_MUNICIPIOS.items():
        d = float(distancia_km(lat, lon, mlat, mlon))
        if mejor_d is None or d < mejor_d:
            mejor_d = d
            mejor = mun
    return mejor


def region_from_point(lat: float, lon: float) -> str:
    mun = municipio_mas_cercano(lat, lon)
    return REGION_MUNICIPIO.get(mun, "SIN_REGION")


def compute_capacities(n: int, entrevistadores: list[str], min_per: int, max_per: int) -> dict[str, int]:
    k = len(entrevistadores)
    if k == 0:
        return {}

    n_asignable = min(n, k * max_per)
    base = n_asignable // k
    residuo = n_asignable % k

    caps = {}
    for i, ent in enumerate(entrevistadores):
        cap = base + (1 if i < residuo else 0)
        cap = max(cap, min_per)
        cap = min(cap, max_per)
        caps[ent] = cap

    total = sum(caps.values())

    if total > n_asignable:
        excedente = total - n_asignable
        for ent in reversed(entrevistadores):
            reducible = max(caps[ent] - min_per, 0)
            delta = min(reducible, excedente)
            caps[ent] -= delta
            excedente -= delta
            if excedente == 0:
                break
    elif total < n_asignable:
        faltante = n_asignable - total
        for ent in entrevistadores:
            expandible = max(max_per - caps[ent], 0)
            delta = min(expandible, faltante)
            caps[ent] += delta
            faltante -= delta
            if faltante == 0:
                break

    return caps


def build_knn_indices(coords: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    n = len(coords)
    if n <= 1:
        return np.zeros((n, 0), dtype=int)

    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    k_use = min(k_neighbors, n - 1)
    return np.argsort(d2, axis=1)[:, :k_use]


# ============================================================
# Geometría para mapa
# ============================================================
def point_in_polygon(lat: float, lon: float, polygon_latlon: list[tuple[float, float]]) -> bool:
    if len(polygon_latlon) < 3 or pd.isna(lat) or pd.isna(lon):
        return False

    x = lon
    y = lat
    inside = False
    n = len(polygon_latlon)

    for i in range(n):
        y1, x1 = polygon_latlon[i]
        y2, x2 = polygon_latlon[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / ((y2 - y1) + 1e-12) + x1
            if x < xinters:
                inside = not inside
    return inside


def point_in_circle(lat: float, lon: float, center_lat: float, center_lon: float, radius_m: float) -> bool:
    d = float(distancia_km(lat, lon, center_lat, center_lon)) * 1000.0
    return d <= radius_m


def extract_shape_selector(feature: dict):
    if not feature:
        return None

    geom = feature.get("geometry", {})
    props = feature.get("properties", {}) or {}
    gtype = geom.get("type")

    if gtype == "Polygon":
        coords = geom.get("coordinates", [])
        if not coords or not coords[0]:
            return None
        ring = coords[0]
        polygon_latlon = [(float(lat), float(lon)) for lon, lat in ring]
        return lambda lat, lon: point_in_polygon(lat, lon, polygon_latlon)

    if gtype == "Point" and "radius" in props:
        lon, lat = geom.get("coordinates", [None, None])
        radius_m = float(props["radius"])
        if lat is None or lon is None:
            return None
        return lambda plat, plon: point_in_circle(plat, plon, float(lat), float(lon), radius_m)

    return None


# ============================================================
# Algoritmo fuerte de agrupamiento compacto y balanceado
# ============================================================
def compact_balanced_assignment(
    df_validos: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    min_per: int,
    max_per: int,
    priority_col: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Objetivos:
    - mantener cargas equilibradas sin ser rígidamente idénticas
    - priorizar cercanía a radicación
    - formar clústeres compactos
    - reducir fragmentación con penalización vecinal
    - mejorar con relocalizaciones y swaps locales
    """
    df = df_validos.copy().reset_index(drop=True)
    pl = plantilla.copy()

    entrevistadores = ordenar_natural(pl["ENTREVISTADOR"].tolist())
    if len(entrevistadores) == 0:
        raise ValueError("No hay entrevistadores en la plantilla.")

    pl = pl.set_index("ENTREVISTADOR").loc[entrevistadores].reset_index()

    prioridad = prioridad_serie(df, priority_col)
    df["_priority_score"] = prioridad
    df["_region_punto"] = df.apply(lambda r: region_from_point(r[lat_col], r[lon_col]), axis=1)
    pl["_region_base"] = pl["MUNICIPIO_RADICACION"].map(lambda x: REGION_MUNICIPIO.get(x, "SIN_REGION"))

    capacidad_total = len(entrevistadores) * int(max_per)

    df = df.sort_values(
        by=["_priority_score", lat_col, lon_col],
        ascending=[False, True, True],
        kind="stable"
    ).reset_index(drop=True)

    if len(df) > capacidad_total:
        df_asignable = df.iloc[:capacidad_total].copy()
        df_exceso = df.iloc[capacidad_total:].copy()
    else:
        df_asignable = df.copy()
        df_exceso = pd.DataFrame(columns=df.columns)

    n = len(df_asignable)
    k = len(entrevistadores)
    caps = compute_capacities(n, entrevistadores, min_per, max_per)

    if n == 0:
        df_asignable["asignado_a"] = None
        df_asignable["estatus_asignacion"] = "Sin registros"
        return df_asignable, df_exceso

    coords = df_asignable[[lat_col, lon_col]].to_numpy(dtype=float)
    rad = pl[["LAT_RADICACION", "LON_RADICACION"]].to_numpy(dtype=float)

    # Distancia a radicación
    dist_base = np.zeros((n, k), dtype=float)
    for j in range(k):
        dist_base[:, j] = distancia_km(coords[:, 0], coords[:, 1], rad[j, 0], rad[j, 1])

    # kNN para contigüidad
    knn_idx = build_knn_indices(coords, k_neighbors=8)

    # Semilla geográfica
    try:
        n_clusters = min(max(k, 2), min(n, k + 4))
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        seed_labels = km.fit_predict(coords)
        seed_centroids = km.cluster_centers_
    except Exception:
        seed_labels = np.zeros(n, dtype=int)
        seed_centroids = np.array([[coords[:, 0].mean(), coords[:, 1].mean()]])

    # Centroides iniciales por entrevistador:
    # mezcla entre radicación y centroides geográficos más cercanos
    centroides = np.zeros((k, 2), dtype=float)
    seed_taken = set()
    for j in range(k):
        seed_d = np.array([
            float(distancia_km(rad[j, 0], rad[j, 1], c[0], c[1]))
            for c in seed_centroids
        ])
        order = np.argsort(seed_d)
        chosen = None
        for idx in order:
            if idx not in seed_taken:
                chosen = idx
                break
        if chosen is None:
            chosen = int(order[0])
        seed_taken.add(chosen)
        centroides[j] = 0.55 * rad[j] + 0.45 * seed_centroids[chosen]

    # Costos base
    def compute_cost_matrix(assignments: np.ndarray, centroides_local: np.ndarray) -> np.ndarray:
        dist_cent = np.zeros((n, k), dtype=float)
        for j in range(k):
            dist_cent[:, j] = distancia_km(
                coords[:, 0], coords[:, 1],
                centroides_local[j, 0], centroides_local[j, 1]
            )

        cost = np.zeros((n, k), dtype=float)
        for j in range(k):
            rb = pl.loc[j, "_region_base"]
            region_pen = np.where(df_asignable["_region_punto"].to_numpy() == rb, 0.0, 70.0)

            # salto largo fuerte
            long_pen = np.where(
                dist_base[:, j] > 140, 800.0,
                np.where(dist_base[:, j] > 100, 240.0,
                         np.where(dist_base[:, j] > 80, 80.0, 0.0))
            )

            # prioridad: si es importante, castiga más irse lejos
            prio_mult = 1.0 + np.clip(df_asignable["_priority_score"].to_numpy(), 0, 5) * 0.06

            cost[:, j] = (
                0.60 * dist_cent[:, j]
                + 0.40 * dist_base[:, j] * prio_mult
                + region_pen
                + long_pen
            )

        # penalización de fragmentación vecinal según asignación actual
        if assignments is not None and len(knn_idx) > 0:
            for i in range(n):
                neigh = knn_idx[i]
                neigh_asig = assignments[neigh]
                for j in range(k):
                    ent = entrevistadores[j]
                    frac_diff = np.mean(neigh_asig != ent) if len(neigh_asig) > 0 else 0.0
                    cost[i, j] += 22.0 * frac_diff
        return cost

    # Asignación inicial greedy capacitada
    caps_arr = np.array([caps[e] for e in entrevistadores], dtype=int)
    ocup = np.zeros(k, dtype=int)
    assignments = np.array([""] * n, dtype=object)
    cost0 = compute_cost_matrix(None, centroides)

    # Orden: puntos difíciles primero
    spread = np.partition(cost0, 1, axis=1)[:, 1] - np.min(cost0, axis=1) if k > 1 else np.zeros(n)
    order_points = np.lexsort((np.min(cost0, axis=1), spread, -df_asignable["_priority_score"].to_numpy()))

    for i in order_points:
        order_j = np.argsort(cost0[i])
        chosen = None
        for j in order_j:
            if ocup[j] < caps_arr[j]:
                chosen = j
                break
        if chosen is None:
            chosen = int(np.argmin(cost0[i]))
        assignments[i] = entrevistadores[chosen]
        ocup[chosen] += 1

    # Refinamiento iterativo
    ent_to_idx = {e: j for j, e in enumerate(entrevistadores)}

    def recompute_centroids(assignments_local: np.ndarray, centroides_prev: np.ndarray) -> np.ndarray:
        cent = centroides_prev.copy()
        for j, ent in enumerate(entrevistadores):
            mask = assignments_local == ent
            if mask.any():
                pts = coords[mask]
                mean_pt = pts.mean(axis=0)
                base_pt = rad[j]
                # ancla moderada a base para compactar sin olvidar radicación
                cent[j] = 0.78 * mean_pt + 0.22 * base_pt
        return cent

    def total_cost(assignments_local: np.ndarray, centroides_local: np.ndarray) -> float:
        cmat = compute_cost_matrix(assignments_local, centroides_local)
        return float(sum(cmat[i, ent_to_idx[assignments_local[i]]] for i in range(n)))

    centroides = recompute_centroids(assignments, centroides)
    best_assign = assignments.copy()
    best_cent = centroides.copy()
    best_cost = total_cost(best_assign, best_cent)

    # Reasignación + local search
    for _ in range(10):
        centroides = recompute_centroids(assignments, centroides)
        cmat = compute_cost_matrix(assignments, centroides)

        ocup = np.array([(assignments == e).sum() for e in entrevistadores], dtype=int)

        moved = 0
        point_order = np.argsort([
            cmat[i, ent_to_idx[assignments[i]]]
            for i in range(n)
        ])[::-1]

        # Relocalización simple
        for i in point_order:
            cur_e = assignments[i]
            cur_j = ent_to_idx[cur_e]
            cur_cost = cmat[i, cur_j]

            order_j = np.argsort(cmat[i])
            for j in order_j:
                if j == cur_j:
                    continue
                if ocup[j] >= caps_arr[j]:
                    continue
                if cmat[i, j] + 1e-9 < cur_cost - 6.0:
                    assignments[i] = entrevistadores[j]
                    ocup[cur_j] -= 1
                    ocup[j] += 1
                    moved += 1
                    break

        centroides = recompute_centroids(assignments, centroides)
        cmat = compute_cost_matrix(assignments, centroides)

        # Swaps locales
        for _swap_round in range(2):
            changed_swap = 0
            worst_idx = np.argsort([
                cmat[i, ent_to_idx[assignments[i]]]
                for i in range(n)
            ])[::-1][: min(120, n)]

            for i in worst_idx:
                ei = assignments[i]
                ji = ent_to_idx[ei]
                ci = cmat[i, ji]

                for t in knn_idx[i] if len(knn_idx) > 0 else []:
                    et = assignments[t]
                    if et == ei:
                        continue
                    jt = ent_to_idx[et]
                    ct = cmat[t, jt]

                    new_cost = cmat[i, jt] + cmat[t, ji]
                    old_cost = ci + ct

                    if new_cost + 1e-9 < old_cost - 8.0:
                        assignments[i], assignments[t] = assignments[t], assignments[i]
                        changed_swap += 1
                        break

            if changed_swap == 0:
                break

        centroides = recompute_centroids(assignments, centroides)
        cur_cost = total_cost(assignments, centroides)

        if cur_cost + 1e-9 < best_cost:
            best_cost = cur_cost
            best_assign = assignments.copy()
            best_cent = centroides.copy()

        if moved == 0:
            break

    assignments = best_assign
    centroides = best_cent

    df_asignable["asignado_a"] = assignments
    df_asignable["estatus_asignacion"] = "Asignado"
    return df_asignable, df_exceso


def recalcular_metricas_asignacion(
    df: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
) -> pd.DataFrame:
    out = df.copy()

    mapa_sup = dict(zip(plantilla["ENTREVISTADOR"], plantilla["SUPERV"]))
    mapa_roe = dict(zip(plantilla["ENTREVISTADOR"], plantilla["ROE"]))
    mapa_mun = dict(zip(plantilla["ENTREVISTADOR"], plantilla["MUNICIPIO_RADICACION"]))
    mapa_lat_rad = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LAT_RADICACION"]))
    mapa_lon_rad = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LON_RADICACION"]))

    out["SUPERV"] = out["asignado_a"].map(mapa_sup)
    out["ROE"] = out["asignado_a"].map(mapa_roe)
    out["MUNICIPIO_RADICACION"] = out["asignado_a"].map(mapa_mun)
    out["LAT_RADICACION"] = out["asignado_a"].map(mapa_lat_rad)
    out["LON_RADICACION"] = out["asignado_a"].map(mapa_lon_rad)

    centroides = (
        out[out["asignado_a"].notna()]
        .groupby("asignado_a")[[lat_col, lon_col]]
        .mean()
        .to_dict("index")
    )

    out["distancia_centroide"] = out.apply(
        lambda row: float(
            distancia_km(
                row[lat_col],
                row[lon_col],
                centroides[row["asignado_a"]][lat_col],
                centroides[row["asignado_a"]][lon_col],
            )
        )
        if pd.notna(row["asignado_a"]) and row["asignado_a"] in centroides
        else np.nan,
        axis=1,
    )

    out["distancia_radicacion_km"] = out.apply(
        lambda row: float(
            distancia_km(
                row[lat_col],
                row[lon_col],
                row["LAT_RADICACION"],
                row["LON_RADICACION"],
            )
        )
        if pd.notna(row["asignado_a"])
        and pd.notna(row["LAT_RADICACION"])
        and pd.notna(row["LON_RADICACION"])
        else np.nan,
        axis=1,
    )

    out["radicacion_viable"] = out["distancia_radicacion_km"].fillna(np.inf) <= 70
    return out


def generar_alertas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "distancia_centroide" not in out.columns:
        out["distancia_centroide"] = np.nan
    if "distancia_radicacion_km" not in out.columns:
        out["distancia_radicacion_km"] = np.nan
    if "radicacion_viable" not in out.columns:
        out["radicacion_viable"] = False

    out["alerta"] = "OK"
    out.loc[~out["coord_no_nulas"], "alerta"] = "COORDENADAS VACIAS"
    out.loc[out["coord_no_nulas"] & ~out["coord_rango_general"], "alerta"] = "COORDENADAS FUERA DE RANGO"
    out.loc[
        out["coord_no_nulas"] & out["coord_rango_general"] & ~out["coord_sinaloa_aprox"],
        "alerta"
    ] = "REVISAR UBICACION"

    validas = out["coord_valida"] & out["distancia_centroide"].notna()
    if validas.any():
        umbral_disp = out.loc[validas, "distancia_centroide"].quantile(0.98)
        out.loc[validas & (out["distancia_centroide"] > umbral_disp), "alerta"] = "REVISAR DISPERSION"

    out.loc[(out["alerta"] == "OK") & (out["distancia_radicacion_km"] > 70), "alerta"] = "LEJOS DE RADICACION"
    out.loc[(out["alerta"] == "OK") & (out["distancia_radicacion_km"] > 100), "alerta"] = "ASIGNACION CRITICA"
    return out


def preparar_sin_asignar(df: pd.DataFrame, motivo: str) -> pd.DataFrame:
    out = df.copy()
    out["asignado_a"] = None
    out["estatus_asignacion"] = motivo
    out["SUPERV"] = None
    out["ROE"] = None
    out["MUNICIPIO_RADICACION"] = None
    out["LAT_RADICACION"] = np.nan
    out["LON_RADICACION"] = np.nan
    out["distancia_centroide"] = np.nan
    out["distancia_radicacion_km"] = np.nan
    out["radicacion_viable"] = False
    return out


def build_summary(df_resultado: pd.DataFrame, plantilla: pd.DataFrame) -> pd.DataFrame:
    df_asig = df_resultado[df_resultado["asignado_a"].notna()].copy()

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
    resumen["color"] = resumen["entrevistador"].map(color_for_entity)
    resumen["orden_ent"] = resumen["entrevistador"].map(natural_key)
    resumen = resumen.sort_values("orden_ent").drop(columns="orden_ent").reset_index(drop=True)
    return resumen


# ============================================================
# Reasignación interactiva
# ============================================================
def apply_geo_reassignment(
    df: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    shape_feature: dict,
    nuevo_entrevistador: str,
    entrevistadores_visibles: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    selector = extract_shape_selector(shape_feature)
    if selector is None:
        raise ValueError("No se pudo interpretar la figura dibujada. Usa polígono, rectángulo o círculo.")

    out = df.copy()
    mask = out.apply(
        lambda r: bool(selector(r[lat_col], r[lon_col])) if pd.notna(r[lat_col]) and pd.notna(r[lon_col]) else False,
        axis=1,
    )

    if entrevistadores_visibles:
        mask = mask & out["asignado_a"].isin(entrevistadores_visibles)

    out.loc[mask, "asignado_a"] = nuevo_entrevistador
    out.loc[mask, "estatus_asignacion"] = "Asignado (manual mapa)"
    out = recalcular_metricas_asignacion(out, plantilla, lat_col, lon_col)
    out = generar_alertas(out)
    return out, int(mask.sum())


# ============================================================
# Mapa
# ============================================================
def build_folium_map(
    df_resultado: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    id_col: Optional[str],
    roe_filter: str,
    superv_filter: str,
    entrevistadores_focus: list[str],
):
    df = df_resultado.copy()
    pl = plantilla.copy()

    if roe_filter != "Todos":
        df = df[df["ROE"] == roe_filter]
        pl = pl[pl["ROE"] == roe_filter]

    if superv_filter != "Todos":
        df = df[df["SUPERV"] == superv_filter]
        pl = pl[pl["SUPERV"] == superv_filter]

    # automático: si seleccionas entrevistadores, solo ellos aparecen
    if entrevistadores_focus:
        df = df[df["asignado_a"].isin(entrevistadores_focus)]
        pl = pl[pl["ENTREVISTADOR"].isin(entrevistadores_focus)]

    df_map = df[df[lat_col].notna() & df[lon_col].notna()].copy()

    if len(df_map) == 0:
        centro = (24.6, -107.4)
    else:
        centro = (float(df_map[lat_col].mean()), float(df_map[lon_col].mean()))

    m = folium.Map(location=centro, zoom_start=8, tiles="CartoDB positron", control_scale=True)
    Fullscreen().add_to(m)

    draw = Draw(
        export=False,
        position="topleft",
        draw_options={
            "polyline": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
            "rectangle": True,
            "circle": True,
        },
        edit_options={"edit": False, "remove": True},
    )
    draw.add_to(m)

    fg_bases = folium.FeatureGroup(name="Bases de entrevistadores", show=True)
    for row in pl.itertuples():
        color = color_for_entity(row.ENTREVISTADOR)
        popup = (
            f"<b>{row.ENTREVISTADOR}</b><br>"
            f"Supervisor: {row.SUPERV}<br>"
            f"ROE: {row.ROE}<br>"
            f"Radicación: {row.MUNICIPIO_RADICACION}"
        )
        folium.Marker(
            location=[row.LAT_RADICACION, row.LON_RADICACION],
            icon=folium.Icon(color="black", icon="home", prefix="fa"),
            tooltip=f"Base {row.ENTREVISTADOR}",
            popup=popup,
        ).add_to(fg_bases)

        folium.Circle(
            location=[row.LAT_RADICACION, row.LON_RADICACION],
            radius=2500,
            color=color,
            fill=False,
            weight=2,
            opacity=0.95,
        ).add_to(fg_bases)
    fg_bases.add_to(m)

    fg_pts = folium.FeatureGroup(name="Registros", show=True)
    for row in df_map.itertuples():
        ent = getattr(row, "asignado_a", None)
        alerta = getattr(row, "alerta", None)
        color = color_for_entity(ent if pd.notna(ent) else str(alerta))

        id_val = getattr(row, id_col) if (id_col is not None and hasattr(row, id_col)) else row.Index

        popup_html = (
            f"<b>ID:</b> {id_val}<br>"
            f"<b>Entrevistador:</b> {ent}<br>"
            f"<b>Supervisor:</b> {getattr(row, 'SUPERV', None)}<br>"
            f"<b>ROE:</b> {getattr(row, 'ROE', None)}<br>"
            f"<b>Radicación:</b> {getattr(row, 'MUNICIPIO_RADICACION', None)}<br>"
            f"<b>Dist. base:</b> {round(getattr(row, 'distancia_radicacion_km', np.nan), 1) if pd.notna(getattr(row, 'distancia_radicacion_km', np.nan)) else 'NA'} km<br>"
            f"<b>Alerta:</b> {alerta}"
        )

        folium.CircleMarker(
            location=[getattr(row, lat_col), getattr(row, lon_col)],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.95,
            opacity=0.95,
            weight=1.5,
            tooltip=f"{ent if pd.notna(ent) else alerta}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(fg_pts)
    fg_pts.add_to(m)

    fg_labels = folium.FeatureGroup(name="Etiquetas de carga", show=True)
    df_asig = df[df["asignado_a"].notna()].copy()

    cargas = (
        df_asig.groupby("asignado_a")
        .agg(registros=(lat_col, "count"), lat=(lat_col, "mean"), lon=(lon_col, "mean"))
        .reset_index()
    )

    for r in cargas.itertuples():
        color = color_for_entity(r.asignado_a)
        html = f"""
        <div style="
            background-color:{color};
            color:white;
            border:2px solid white;
            border-radius:16px;
            padding:4px 8px;
            font-size:12px;
            font-weight:bold;
            box-shadow:0 0 6px rgba(0,0,0,0.35);
            white-space:nowrap;">
            {r.asignado_a}: {int(r.registros)}
        </div>
        """
        folium.Marker(
            location=[r.lat, r.lon],
            icon=DivIcon(icon_size=(100, 24), icon_anchor=(0, 0), html=html),
        ).add_to(fg_labels)

    fg_labels.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_side_cards(resumen_panel: pd.DataFrame):
    if resumen_panel.empty:
        st.info("No hay entrevistadores en el filtro actual.")
        return

    max_reg = max(int(resumen_panel["registros"].max()), 1)

    for row in resumen_panel.itertuples():
        pct = int(round((row.registros / max_reg) * 100))
        color = row.color
        st.markdown(
            f"""
            <div style="
                border:1px solid #e6e6e6;
                border-left:8px solid {color};
                border-radius:10px;
                padding:10px 12px;
                margin-bottom:10px;
                background:#ffffff;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div style="font-weight:700; font-size:15px;">{row.entrevistador}</div>
                    <div style="font-weight:700; font-size:18px;">{int(row.registros)}</div>
                </div>
                <div style="font-size:12px; color:#555; margin-top:2px;">
                    {row.SUPERV} | {row.ROE} | {row.MUNICIPIO_RADICACION}
                </div>
                <div style="
                    margin-top:8px;
                    width:100%;
                    background:#f0f2f6;
                    border-radius:999px;
                    height:10px;">
                    <div style="
                        width:{pct}%;
                        background:{color};
                        height:10px;
                        border-radius:999px;">
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# Interfaz principal
# ============================================================
st.title("Planeación operativa de entrevistadores")
st.caption(
    "Planeación automática compacta, balanceada y orientada a minimizar traslados, gasolina y fragmentación territorial."
)

uploaded = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info(
        "La aplicación detecta columnas geográficas, genera una planeación compacta y permite redistribuir puntos directamente sobre el mapa."
    )
    st.stop()

file_bytes = uploaded.getvalue()
sheets = load_excel(file_bytes, uploaded.name)

if not sheets:
    st.error("No fue posible leer hojas del archivo.")
    st.stop()

st.sidebar.header("Configuración base")
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

st.sidebar.subheader("Capacidad operativa")
num_interviewers = st.sidebar.number_input("Número de entrevistadores", min_value=1, value=6, step=1)
num_supervisores = st.sidebar.number_input("Supervisores", min_value=1, value=3, step=1)
min_per = st.sidebar.number_input("Mínimo por entrevistador", min_value=0, value=80, step=1)
max_per = st.sidebar.number_input("Máximo por entrevistador", min_value=1, value=115, step=1)

if not lat_col or not lon_col:
    st.error("Selecciona las columnas de latitud y longitud para continuar.")
    st.stop()

if int(min_per) > int(max_per):
    st.error("El mínimo por entrevistador no puede ser mayor que el máximo.")
    st.stop()

with st.expander("Plantilla operativa", expanded=True):
    plantilla_default = construir_plantilla(int(num_interviewers), int(num_supervisores))
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
    plantilla_edit["LAT_RADICACION"] = plantilla_edit["MUNICIPIO_RADICACION"].map(lambda x: COORD_MUNICIPIOS[x][0])
    plantilla_edit["LON_RADICACION"] = plantilla_edit["MUNICIPIO_RADICACION"].map(lambda x: COORD_MUNICIPIOS[x][1])

df_geo = clean_geo(df_raw, lat_col, lon_col)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Registros en hoja", len(df_raw))
c2.metric("Coordenadas válidas", int(df_geo["coord_valida"].sum()))
c3.metric("Sin coordenadas", int(df_geo[[lat_col, lon_col]].isna().any(axis=1).sum()))
c4.metric("Entrevistadores", len(plantilla_edit))

with st.expander("Vista previa del archivo", expanded=False):
    st.dataframe(df_raw.head(30), use_container_width=True)

run = st.button("Generar planeación automática", type="primary")

if run:
    try:
        df_work = df_geo.copy()
        df_work["row_id"] = np.arange(len(df_work))

        validos = df_work[df_work["coord_valida"]].copy()
        invalidos = df_work[~df_work["coord_valida"]].copy()

        if len(validos) < len(plantilla_edit):
            st.error("No hay suficientes registros válidos para asignar al menos un registro por entrevistador.")
            st.stop()

        asignados, exceso = compact_balanced_assignment(
            validos,
            plantilla_edit,
            lat_col=lat_col,
            lon_col=lon_col,
            min_per=int(min_per),
            max_per=int(max_per),
            priority_col=priority_col,
        )

        asignados = recalcular_metricas_asignacion(asignados, plantilla_edit, lat_col, lon_col)
        asignados = generar_alertas(asignados)

        piezas = [asignados]

        if len(exceso) > 0:
            exceso = preparar_sin_asignar(exceso, "Sin asignar por capacidad máxima")
            exceso["alerta"] = "SIN CAPACIDAD"
            piezas.append(exceso)

        if len(invalidos) > 0:
            invalidos = preparar_sin_asignar(invalidos, "Coordenada inválida o a revisar")
            invalidos = generar_alertas(invalidos)
            piezas.append(invalidos)

        resultado = pd.concat(piezas, ignore_index=True)
        st.session_state["resultado_planeacion"] = resultado
        st.session_state["plantilla_planeacion"] = plantilla_edit.copy()
        st.success("Planeación generada correctamente.")
        st.rerun()

    except Exception as e:
        st.exception(e)

if "resultado_planeacion" not in st.session_state or "plantilla_planeacion" not in st.session_state:
    st.stop()

resultado = st.session_state["resultado_planeacion"].copy()
plantilla_vigente = st.session_state["plantilla_planeacion"].copy()

st.subheader("Mapa principal de planeación")

flt1, flt2, flt3 = st.columns([1, 1, 1.6])

roe_options = ["Todos"] + ordenar_natural(plantilla_vigente["ROE"].dropna().unique().tolist())
roe_filter = flt1.selectbox("Filtrar por ROE", roe_options, index=0)

if roe_filter == "Todos":
    sup_base = plantilla_vigente.copy()
else:
    sup_base = plantilla_vigente[plantilla_vigente["ROE"] == roe_filter].copy()

sup_options = ["Todos"] + ordenar_natural(sup_base["SUPERV"].dropna().unique().tolist())
superv_filter = flt2.selectbox("Filtrar por supervisor", sup_options, index=0)

ent_base = plantilla_vigente.copy()
if roe_filter != "Todos":
    ent_base = ent_base[ent_base["ROE"] == roe_filter]
if superv_filter != "Todos":
    ent_base = ent_base[ent_base["SUPERV"] == superv_filter]

ent_options = ordenar_natural(ent_base["ENTREVISTADOR"].dropna().unique().tolist())
entrevistadores_focus = flt3.multiselect(
    "Seleccionar entrevistadores",
    ent_options,
    default=[],
)

resumen = build_summary(resultado, plantilla_vigente)

left_map, right_panel = st.columns([3.6, 1.1], gap="large")

with left_map:
    mapa = build_folium_map(
        resultado,
        plantilla_vigente,
        lat_col,
        lon_col,
        id_col,
        roe_filter,
        superv_filter,
        entrevistadores_focus,
    )

    map_state = st_folium(
        mapa,
        height=760,
        width=None,
        key="mapa_planeacion",
        returned_objects=["last_active_drawing", "all_drawings"],
    )

with right_panel:
    st.markdown("### Cargas por entrevistador")

    resumen_panel = resumen.copy()
    if roe_filter != "Todos":
        resumen_panel = resumen_panel[resumen_panel["ROE"] == roe_filter]
    if superv_filter != "Todos":
        resumen_panel = resumen_panel[resumen_panel["SUPERV"] == superv_filter]
    if entrevistadores_focus:
        resumen_panel = resumen_panel[resumen_panel["entrevistador"].isin(entrevistadores_focus)]

    total_asignados = int(resultado["asignado_a"].notna().sum())
    total_sin_asignar = int(resultado["asignado_a"].isna().sum())
    st.metric("Asignados", total_asignados)
    st.metric("Sin asignar", total_sin_asignar)

    render_side_cards(resumen_panel)

    st.markdown("### Redistribución directa")
    opciones_target = ent_options if ent_options else ordenar_natural(plantilla_vigente["ENTREVISTADOR"].tolist())
    target_ent = st.selectbox(
        "Mover puntos al entrevistador",
        opciones_target,
        key="target_ent_mapa",
    )

    limitar_a_foco = st.checkbox(
        "Reasignar solo puntos de los entrevistadores seleccionados",
        value=True if entrevistadores_focus else False,
    )

    st.caption("Dibuja en el mapa un polígono, rectángulo o círculo y aplica la reasignación.")

    if st.button("Aplicar reasignación de la figura", type="secondary"):
        try:
            drawing = None
            if map_state is not None:
                drawing = map_state.get("last_active_drawing")
                if drawing is None:
                    drawings = map_state.get("all_drawings")
                    if drawings and isinstance(drawings, list):
                        drawing = drawings[-1]

            if drawing is None:
                st.warning("No hay figura seleccionada en el mapa.")
            else:
                visibles = entrevistadores_focus if (limitar_a_foco and entrevistadores_focus) else None
                nuevo_resultado, movidos = apply_geo_reassignment(
                    resultado,
                    plantilla_vigente,
                    lat_col,
                    lon_col,
                    drawing,
                    target_ent,
                    entrevistadores_visibles=visibles,
                )
                st.session_state["resultado_planeacion"] = nuevo_resultado
                st.success(f"Reasignación aplicada. Registros movidos: {movidos}.")
                st.rerun()
        except Exception as e:
            st.error(f"No se pudo aplicar la reasignación: {e}")

    if st.button("Restablecer a la última planeación automática"):
        try:
            df_work = df_geo.copy()
            df_work["row_id"] = np.arange(len(df_work))

            validos = df_work[df_work["coord_valida"]].copy()
            invalidos = df_work[~df_work["coord_valida"]].copy()

            asignados, exceso = compact_balanced_assignment(
                validos,
                plantilla_vigente,
                lat_col=lat_col,
                lon_col=lon_col,
                min_per=int(min_per),
                max_per=int(max_per),
                priority_col=priority_col,
            )

            asignados = recalcular_metricas_asignacion(asignados, plantilla_vigente, lat_col, lon_col)
            asignados = generar_alertas(asignados)

            piezas = [asignados]

            if len(exceso) > 0:
                exceso = preparar_sin_asignar(exceso, "Sin asignar por capacidad máxima")
                exceso["alerta"] = "SIN CAPACIDAD"
                piezas.append(exceso)

            if len(invalidos) > 0:
                invalidos = preparar_sin_asignar(invalidos, "Coordenada inválida o a revisar")
                invalidos = generar_alertas(invalidos)
                piezas.append(invalidos)

            resultado_reset = pd.concat(piezas, ignore_index=True)
            st.session_state["resultado_planeacion"] = resultado_reset
            st.success("Planeación restablecida.")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo restablecer la planeación: {e}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Asignados", int(resultado["asignado_a"].notna().sum()))
k2.metric("Sin asignar", int(resultado["asignado_a"].isna().sum()))
k3.metric(
    "Distancia media a radicación",
    round(resultado["distancia_radicacion_km"].dropna().mean(), 1)
    if "distancia_radicacion_km" in resultado.columns and resultado["distancia_radicacion_km"].notna().any()
    else 0
)
k4.metric(
    "Asignaciones críticas",
    int((resultado["alerta"] == "ASIGNACION CRITICA").sum())
    if "alerta" in resultado.columns else 0
)

tab1, tab2, tab3 = st.tabs(["Resumen operativo", "Alertas", "Detalle de registros"])

with tab1:
    st.dataframe(resumen, use_container_width=True)

with tab2:
    inconsistencias = resultado[resultado["alerta"] != "OK"].copy() if "alerta" in resultado.columns else pd.DataFrame()
    st.dataframe(inconsistencias, use_container_width=True)

with tab3:
    st.dataframe(resultado, use_container_width=True, height=520)

export_bytes = dataframe_to_excel_bytes(
    {
        "asignacion": resultado,
        "resumen_entrevistador": resumen,
        "plantilla": plantilla_vigente,
        "parametros": pd.DataFrame(
            {
                "parametro": [
                    "hoja_seleccionada",
                    "lat_col",
                    "lon_col",
                    "id_col",
                    "priority_col",
                    "entrevistadores",
                    "supervisores",
                    "min_por_entrevistador",
                    "max_por_entrevistador",
                ],
                "valor": [
                    selected_sheet,
                    lat_col,
                    lon_col,
                    id_col,
                    priority_col,
                    len(plantilla_vigente),
                    plantilla_vigente["SUPERV"].nunique(),
                    min_per,
                    max_per,
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

st.markdown("---")
st.markdown(
    """
    **Mejoras incluidas en esta versión**
    - Agrupamiento más compacto y balanceado por entrevistador.
    - Penalización por fragmentación para evitar cargas mal partidas.
    - Reasignación automática del mapa a solo los entrevistadores seleccionados.
    - Refinamiento por relocalizaciones y swaps para reducir traslados.
    """
)
