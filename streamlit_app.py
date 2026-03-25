import re
import zipfile
from io import BytesIO
from typing import Optional, Tuple
from xml.sax.saxutils import escape

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.features import DivIcon
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

st.set_page_config(page_title="Planeación operativa - INEGI - 2026", layout="wide")

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

WEEK_COLS = ["SEMANA_RECORRIDO", "SEMANA_NUM", "ORDEN_SEMANA"]


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


def soft_targets(n: int, entrevistadores: list[str], min_per: int, max_per: int) -> dict[str, int]:
    k = len(entrevistadores)
    if k == 0:
        return {}

    ideal = n / k
    base = int(np.floor(ideal))
    residuo = int(n - base * k)

    targets = {}
    for i, ent in enumerate(entrevistadores):
        val = base + (1 if i < residuo else 0)
        val = min(val, max_per)
        targets[ent] = max(val, min(min_per, max_per, max(0, int(np.floor(ideal)))))
    return targets


def build_knn_indices(coords: np.ndarray, k_neighbors: int = 8) -> np.ndarray:
    n = len(coords)
    if n <= 1:
        return np.zeros((n, 0), dtype=int)

    diffs = coords[:, None, :] - coords[None, :, :]
    d2 = (diffs ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    k_use = min(k_neighbors, n - 1)
    return np.argsort(d2, axis=1)[:, :k_use]


def standardize_xy(coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return coords.copy()
    mu = coords.mean(axis=0)
    sd = coords.std(axis=0)
    sd[sd == 0] = 1.0
    return (coords - mu) / sd


def latlon_to_local_xy(lat: np.ndarray, lon: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    ref_lat_r = np.radians(ref_lat)
    ref_lon_r = np.radians(ref_lon)

    x = (lon_r - ref_lon_r) * np.cos((lat_r + ref_lat_r) / 2.0) * 6371.0
    y = (lat_r - ref_lat_r) * 6371.0
    return np.column_stack([x, y])


def group_indices_by_base(rad: np.ndarray, decimals: int = 5) -> dict[tuple[float, float], list[int]]:
    groups = {}
    for j, (la, lo) in enumerate(rad):
        key = (round(float(la), decimals), round(float(lo), decimals))
        groups.setdefault(key, []).append(j)
    return groups


def partition_contiguous(order_idx: np.ndarray, quotas: list[int]) -> list[np.ndarray]:
    parts = []
    start = 0
    n = len(order_idx)
    for i, q in enumerate(quotas):
        if i == len(quotas) - 1:
            end = n
        else:
            end = start + int(q)
        parts.append(order_idx[start:end])
        start = end
    return parts


def contiguous_sector_assignment(
    coords_group: np.ndarray,
    ent_names_group: list[str],
    quota_map: dict[str, int],
    base_lat: float,
    base_lon: float,
) -> dict[int, str]:
    n = len(coords_group)
    if n == 0:
        return {}

    if len(ent_names_group) == 1:
        return {i: ent_names_group[0] for i in range(n)}

    local_xy = latlon_to_local_xy(coords_group[:, 0], coords_group[:, 1], base_lat, base_lon)
    ang = np.mod(np.arctan2(local_xy[:, 1], local_xy[:, 0]), 2 * np.pi)
    rad = np.sqrt((local_xy ** 2).sum(axis=1))

    ent_sorted = ordenar_natural(ent_names_group)
    quotas = [int(quota_map[e]) for e in ent_sorted]

    order0 = np.lexsort((rad, ang))
    best_cost = None
    best_parts = None

    n_trials = min(24, max(n, 1))
    starts = np.linspace(0, n - 1, n_trials, dtype=int) if n > 0 else np.array([0], dtype=int)

    for s in starts:
        rolled = np.roll(order0, -int(s))
        parts = partition_contiguous(rolled, quotas)

        cost = 0.0
        for part in parts:
            if len(part) <= 1:
                continue
            pts = local_xy[part]
            centro = pts.mean(axis=0)
            cost += float(((pts - centro) ** 2).sum())

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_parts = parts

    assignment_local = {}
    for ent, part in zip(ent_sorted, best_parts):
        for idx_local in part:
            assignment_local[int(idx_local)] = ent

    return assignment_local


def quotas_for_group(n_points: int, ent_names_group: list[str], soft_target_map: dict[str, int], max_per: int) -> dict[str, int]:
    if len(ent_names_group) == 0:
        return {}

    hard_caps = {e: max_per for e in ent_names_group}
    weights = np.array([max(soft_target_map.get(e, 1), 1) for e in ent_names_group], dtype=float)
    weights = weights / weights.sum()

    raw = weights * n_points
    floors = np.floor(raw).astype(int)
    resid = int(n_points - floors.sum())

    quotas = {e: int(f) for e, f in zip(ent_names_group, floors)}

    if resid > 0:
        frac = raw - floors
        order = np.argsort(frac)[::-1]
        for idx in order[:resid]:
            quotas[ent_names_group[int(idx)]] += 1

    for e in ent_names_group:
        quotas[e] = min(quotas[e], hard_caps[e])

    total = sum(quotas.values())
    if total < n_points:
        faltan = n_points - total
        for e in ent_names_group:
            disp = hard_caps[e] - quotas[e]
            add = min(disp, faltan)
            quotas[e] += add
            faltan -= add
            if faltan == 0:
                break

    return quotas


# ============================================================
# Planeación semanal y exportación Google Earth
# ============================================================
def drop_week_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols_drop = [c for c in WEEK_COLS if c in out.columns]
    if cols_drop:
        out = out.drop(columns=cols_drop)
    return out


def even_quotas(n: int, k: int) -> list[int]:
    if k <= 0:
        return []
    base = n // k
    resid = n % k
    return [base + (1 if i < resid else 0) for i in range(k)]


def assign_weeks_compact_for_interviewer(
    sub: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    base_lat: float,
    base_lon: float,
    n_weeks: int,
) -> pd.DataFrame:
    out = sub.copy().reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    n = len(out)

    if n == 0:
        out["SEMANA_NUM"] = np.nan
        out["SEMANA_RECORRIDO"] = None
        out["ORDEN_SEMANA"] = np.nan
        return out

    n_weeks_use = max(1, min(int(n_weeks), n))
    coords = out[[lat_col, lon_col]].to_numpy(dtype=float)

    local_xy = latlon_to_local_xy(
        coords[:, 0],
        coords[:, 1],
        float(base_lat),
        float(base_lon),
    )
    ang = np.mod(np.arctan2(local_xy[:, 1], local_xy[:, 0]), 2 * np.pi)
    rad = np.sqrt((local_xy ** 2).sum(axis=1))

    order0 = np.lexsort((rad, ang))
    quotas = even_quotas(n, n_weeks_use)

    best_cost = None
    best_parts = None

    n_trials = min(max(n, 1), 24)
    starts = np.linspace(0, max(n - 1, 0), n_trials, dtype=int)

    for s in starts:
        rolled = np.roll(order0, -int(s))
        parts = []
        start = 0
        for i, q in enumerate(quotas):
            end = start + q if i < len(quotas) - 1 else n
            parts.append(rolled[start:end])
            start = end

        cost = 0.0
        for part in parts:
            if len(part) <= 1:
                continue
            pts = local_xy[part]
            centro = pts.mean(axis=0)
            cost += float(((pts - centro) ** 2).sum())

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_parts = parts

    semana_num = np.zeros(n, dtype=int)
    orden_semana = np.zeros(n, dtype=int)

    for w, part in enumerate(best_parts, start=1):
        if len(part) == 0:
            continue

        pts = local_xy[part]
        rad_part = np.sqrt((pts ** 2).sum(axis=1))
        ang_part = np.mod(np.arctan2(pts[:, 1], pts[:, 0]), 2 * np.pi)

        local_order = np.lexsort((rad_part, ang_part))
        ordered_idx = np.array(part)[local_order]

        for pos, idx_local in enumerate(ordered_idx, start=1):
            semana_num[idx_local] = w
            orden_semana[idx_local] = pos

    out["SEMANA_NUM"] = semana_num
    out["SEMANA_RECORRIDO"] = out["SEMANA_NUM"].map(lambda x: f"Semana {int(x)}" if pd.notna(x) and x > 0 else None)
    out["ORDEN_SEMANA"] = orden_semana

    out = out.sort_values(["SEMANA_NUM", "ORDEN_SEMANA"]).reset_index(drop=True)
    return out


def assign_route_weeks(
    df_resultado: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    n_weeks: int,
) -> pd.DataFrame:
    out = drop_week_columns(df_resultado.copy())

    if "asignado_a" not in out.columns:
        return out

    asignados = out[out["asignado_a"].notna()].copy()

    if len(asignados) == 0:
        out["SEMANA_NUM"] = np.nan
        out["SEMANA_RECORRIDO"] = None
        out["ORDEN_SEMANA"] = np.nan
        return out

    mapa_lat = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LAT_RADICACION"]))
    mapa_lon = dict(zip(plantilla["ENTREVISTADOR"], plantilla["LON_RADICACION"]))

    piezas = []

    for ent in ordenar_natural(asignados["asignado_a"].dropna().unique().tolist()):
        sub = asignados[asignados["asignado_a"] == ent].copy()
        base_lat = mapa_lat.get(ent, sub[lat_col].mean())
        base_lon = mapa_lon.get(ent, sub[lon_col].mean())

        sub_week = assign_weeks_compact_for_interviewer(
            sub=sub,
            lat_col=lat_col,
            lon_col=lon_col,
            base_lat=base_lat,
            base_lon=base_lon,
            n_weeks=n_weeks,
        )
        piezas.append(sub_week)

    asignados_week = pd.concat(piezas, ignore_index=True)

    if "_orig_idx" in asignados_week.columns:
        asignados_week = asignados_week.set_index("_orig_idx")

    out = out.reset_index(drop=False).rename(columns={"index": "_orig_idx"}).set_index("_orig_idx")

    for col in ["SEMANA_NUM", "SEMANA_RECORRIDO", "ORDEN_SEMANA"]:
        out[col] = np.nan if col != "SEMANA_RECORRIDO" else None

    for col in ["SEMANA_NUM", "SEMANA_RECORRIDO", "ORDEN_SEMANA"]:
        out.loc[asignados_week.index, col] = asignados_week[col]

    out = out.reset_index(drop=True)
    return out


def build_week_summary(df_resultado: pd.DataFrame) -> pd.DataFrame:
    if "SEMANA_RECORRIDO" not in df_resultado.columns:
        return pd.DataFrame()

    df = df_resultado[df_resultado["asignado_a"].notna() & df_resultado["SEMANA_RECORRIDO"].notna()].copy()
    if len(df) == 0:
        return pd.DataFrame()

    resumen = (
        df.groupby(["asignado_a", "SUPERV", "ROE", "MUNICIPIO_RADICACION", "SEMANA_NUM", "SEMANA_RECORRIDO"], dropna=False)
        .size()
        .reset_index(name="registros")
        .rename(columns={"asignado_a": "entrevistador"})
        .sort_values(["entrevistador", "SEMANA_NUM"])
        .reset_index(drop=True)
    )
    return resumen


def kml_color_from_hex(hex_color: str, alpha: str = "ff") -> str:
    c = (hex_color or "#808080").replace("#", "").strip()
    if len(c) != 6:
        c = "808080"
    rr = c[0:2]
    gg = c[2:4]
    bb = c[4:6]
    return f"{alpha}{bb}{gg}{rr}"


def safe_html_value(v) -> str:
    if pd.isna(v):
        return ""
    return escape(str(v))


def build_kml_description(row: pd.Series, fields: list[str]) -> str:
    rows = []
    for col in fields:
        if col in row.index:
            rows.append(
                f"<tr><td><b>{escape(str(col))}</b></td><td>{safe_html_value(row[col])}</td></tr>"
            )

    html = (
        "<![CDATA["
        "<div style='font-family:Arial,sans-serif;font-size:12px;'>"
        "<table border='1' cellspacing='0' cellpadding='4' style='border-collapse:collapse;'>"
        + "".join(rows) +
        "</table>"
        "</div>"
        "]]>"
    )
    return html


def dataframe_to_kml_string(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    id_col: Optional[str],
    popup_fields: list[str],
) -> str:
    df_exp = df[df[lat_col].notna() & df[lon_col].notna()].copy()

    style_map = {}
    ents = ordenar_natural(df_exp["asignado_a"].fillna("SIN_ASIGNAR").astype(str).unique().tolist())
    for ent in ents:
        color = color_for_entity(ent) if ent != "SIN_ASIGNAR" else "#808080"
        style_map[ent] = kml_color_from_hex(color)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<kml xmlns="http://www.opengis.net/kml/2.2">')
    lines.append("<Document>")
    lines.append("<name>Planeacion operativa</name>")

    for ent in ents:
        style_id = f"style_{re.sub(r'[^A-Za-z0-9_]+', '_', str(ent))}"
        kml_color = style_map[ent]
        lines.append(f'<Style id="{style_id}">')
        lines.append("<IconStyle>")
        lines.append(f"<color>{kml_color}</color>")
        lines.append("<scale>1.1</scale>")
        lines.append("<Icon><href>http://maps.google.com/mapfiles/kml/shapes/shaded_dot.png</href></Icon>")
        lines.append("</IconStyle>")
        lines.append("<LabelStyle><scale>0</scale></LabelStyle>")
        lines.append("</Style>")

    for ent in ents:
        style_id = f"style_{re.sub(r'[^A-Za-z0-9_]+', '_', str(ent))}"
        sub = df_exp[df_exp["asignado_a"].fillna("SIN_ASIGNAR").astype(str) == ent].copy()

        lines.append("<Folder>")
        lines.append(f"<name>{escape(str(ent))}</name>")

        for _, row in sub.iterrows():
            lat = float(row[lat_col])
            lon = float(row[lon_col])

            if id_col is not None and id_col in row.index and pd.notna(row[id_col]):
                point_name = str(row[id_col])
            else:
                point_name = f"{ent}"

            desc = build_kml_description(row, popup_fields)

            lines.append("<Placemark>")
            lines.append(f"<name>{escape(point_name)}</name>")
            lines.append(f"<styleUrl>#{style_id}</styleUrl>")
            lines.append(f"<description>{desc}</description>")
            lines.append("<Point>")
            lines.append(f"<coordinates>{lon},{lat},0</coordinates>")
            lines.append("</Point>")
            lines.append("</Placemark>")

        lines.append("</Folder>")

    lines.append("</Document>")
    lines.append("</kml>")
    return "\n".join(lines)


def dataframe_to_kmz_bytes(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    id_col: Optional[str],
    popup_fields: list[str],
) -> bytes:
    kml_text = dataframe_to_kml_string(
        df=df,
        lat_col=lat_col,
        lon_col=lon_col,
        id_col=id_col,
        popup_fields=popup_fields,
    )

    output = BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_text.encode("utf-8"))
    output.seek(0)
    return output.read()


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
# Algoritmo definitivo: radicación dura + compactación
# ============================================================
def compact_balanced_assignment(
    df_validos: pd.DataFrame,
    plantilla: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    min_per: int,
    max_per: int,
    priority_col: Optional[str] = None,
    hard_max_km: float = 120.0,
    slack_nearest_km: float = 25.0,
    cross_region_max_km: float = 45.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_validos.copy().reset_index(drop=True)
    pl = plantilla.copy()

    entrevistadores = ordenar_natural(pl["ENTREVISTADOR"].tolist())
    if len(entrevistadores) == 0:
        raise ValueError("No hay entrevistadores en la plantilla.")

    pl = pl.set_index("ENTREVISTADOR").loc[entrevistadores].reset_index()

    df["_priority_score"] = prioridad_serie(df, priority_col).astype(float)
    df["_region_punto"] = df.apply(lambda r: region_from_point(r[lat_col], r[lon_col]), axis=1)
    pl["_region_base"] = pl["MUNICIPIO_RADICACION"].map(lambda x: REGION_MUNICIPIO.get(x, "SIN_REGION"))

    n = len(df)
    if n == 0:
        df["asignado_a"] = None
        df["estatus_asignacion"] = "Sin registros"
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()

    capacidad_total = len(entrevistadores) * int(max_per)

    coords = df[[lat_col, lon_col]].to_numpy(dtype=float)
    rad_ent = pl[["LAT_RADICACION", "LON_RADICACION"]].to_numpy(dtype=float)

    dist_ent_all = np.zeros((n, len(entrevistadores)), dtype=float)
    for j in range(len(entrevistadores)):
        dist_ent_all[:, j] = distancia_km(coords[:, 0], coords[:, 1], rad_ent[j, 0], rad_ent[j, 1])

    nearest_any = dist_ent_all.min(axis=1)

    df = df.assign(_nearest_base_km=nearest_any)
    df = df.sort_values(
        by=["_priority_score", "_nearest_base_km"],
        ascending=[False, False],
        kind="stable"
    ).reset_index(drop=True)

    if len(df) > capacidad_total:
        df_asignable = df.iloc[:capacidad_total].copy()
        df_exceso_cap = df.iloc[capacidad_total:].copy()
        df_exceso_cap["asignado_a"] = None
        df_exceso_cap["estatus_asignacion"] = "Sin asignar por capacidad máxima"
    else:
        df_asignable = df.copy()
        df_exceso_cap = df.iloc[0:0].copy()

    if len(df_asignable) == 0:
        return df_asignable, df_exceso_cap

    soft_target_map = soft_targets(len(df_asignable), entrevistadores, min_per, max_per)
    hard_cap_map = {e: int(max_per) for e in entrevistadores}

    rad = pl[["LAT_RADICACION", "LON_RADICACION"]].to_numpy(dtype=float)
    base_groups = group_indices_by_base(rad, decimals=5)

    group_keys = list(base_groups.keys())
    group_meta = []

    for g_idx, key in enumerate(group_keys):
        ent_js = base_groups[key]
        ent_names = [entrevistadores[j] for j in ent_js]
        base_lat, base_lon = key
        hard_cap_total = sum(hard_cap_map[e] for e in ent_names)
        soft_target_total = sum(soft_target_map.get(e, 0) for e in ent_names)
        region_base = REGION_MUNICIPIO.get(pl.loc[ent_js[0], "MUNICIPIO_RADICACION"], "SIN_REGION")

        group_meta.append(
            {
                "group_idx": g_idx,
                "key": key,
                "ent_js": ent_js,
                "ent_names": ent_names,
                "base_lat": float(base_lat),
                "base_lon": float(base_lon),
                "hard_cap_total": int(hard_cap_total),
                "soft_target_total": int(soft_target_total),
                "region_base": region_base,
            }
        )

    n_asig = len(df_asignable)
    coords_asig = df_asignable[[lat_col, lon_col]].to_numpy(dtype=float)
    point_regions = df_asignable["_region_punto"].to_numpy(dtype=object)
    priorities = df_asignable["_priority_score"].to_numpy(dtype=float)

    gcount = len(group_meta)
    dist_group = np.zeros((n_asig, gcount), dtype=float)

    for g, meta in enumerate(group_meta):
        dist_group[:, g] = distancia_km(
            coords_asig[:, 0],
            coords_asig[:, 1],
            meta["base_lat"],
            meta["base_lon"],
        )

    nearest_group_dist = dist_group.min(axis=1)

    feasible = np.zeros((n_asig, gcount), dtype=bool)

    for i in range(n_asig):
        dmin = nearest_group_dist[i]
        for g, meta in enumerate(group_meta):
            d = dist_group[i, g]
            same_region = point_regions[i] == meta["region_base"]

            if d > hard_max_km:
                continue
            if d > dmin + slack_nearest_km:
                continue
            if (not same_region) and d > cross_region_max_km:
                continue

            feasible[i, g] = True

    group_loads = np.zeros(gcount, dtype=int)
    point_to_group = np.full(n_asig, -1, dtype=int)

    order_points = np.lexsort((
        nearest_group_dist,
        -priorities,
    ))

    for i in order_points:
        cand_groups = np.where(feasible[i])[0]

        if len(cand_groups) == 0:
            continue

        best_g = None
        best_cost = None

        for g in cand_groups:
            meta = group_meta[g]

            if group_loads[g] >= meta["hard_cap_total"]:
                continue

            d = dist_group[i, g]
            load_ratio = group_loads[g] / max(meta["hard_cap_total"], 1)

            region_pen = 0.0 if point_regions[i] == meta["region_base"] else 70.0
            prio_mult = 1.0 + np.clip(priorities[i], 0, 5) * 0.08

            dist_pen = 0.0
            if d > 90:
                dist_pen += 120.0 + (d - 90) * 3.0
            elif d > 70:
                dist_pen += 35.0 + (d - 70) * 1.6
            elif d > 45:
                dist_pen += (d - 45) * 0.8

            soft_over_pen = 18.0 * (load_ratio ** 2.0)

            soft_target_total = max(meta["soft_target_total"], 1)
            soft_ratio = group_loads[g] / soft_target_total
            soft_balance_pen = 10.0 * max(soft_ratio - 1.0, 0.0) ** 2

            cost = (
                d * prio_mult
                + region_pen
                + dist_pen
                + soft_over_pen
                + soft_balance_pen
            )

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_g = g

        if best_g is not None:
            point_to_group[i] = best_g
            group_loads[best_g] += 1

    assigned_mask = point_to_group >= 0
    df_group_assigned = df_asignable.loc[assigned_mask].copy().reset_index(drop=False).rename(columns={"index": "_orig_idx"})
    df_group_unassigned = df_asignable.loc[~assigned_mask].copy()

    if len(df_group_unassigned) > 0:
        df_group_unassigned["asignado_a"] = None
        df_group_unassigned["estatus_asignacion"] = "Sin radicación viable"

    final_assignments = pd.Series(index=df_group_assigned.index, dtype=object)

    for g, meta in enumerate(group_meta):
        idx_global = np.where(point_to_group == g)[0]
        if len(idx_global) == 0:
            continue

        sub = df_asignable.iloc[idx_global].copy().reset_index(drop=True)
        coords_group = sub[[lat_col, lon_col]].to_numpy(dtype=float)

        ent_names_group = meta["ent_names"]
        quota_map = quotas_for_group(
            n_points=len(sub),
            ent_names_group=ent_names_group,
            soft_target_map=soft_target_map,
            max_per=max_per,
        )

        local_assign = contiguous_sector_assignment(
            coords_group=coords_group,
            ent_names_group=ent_names_group,
            quota_map=quota_map,
            base_lat=meta["base_lat"],
            base_lon=meta["base_lon"],
        )

        for local_i, ent in local_assign.items():
            orig_i = idx_global[local_i]
            rows = df_group_assigned.index[df_group_assigned["_orig_idx"] == orig_i].tolist()
            if rows:
                final_assignments.loc[rows[0]] = ent

    df_group_assigned["asignado_a"] = final_assignments.values
    df_group_assigned["estatus_asignacion"] = "Asignado"

    still_na = df_group_assigned["asignado_a"].isna()
    if still_na.any():
        extra_na = df_group_assigned.loc[still_na].copy()
        extra_na["estatus_asignacion"] = "Sin asignar por sectorización"
        df_group_assigned = df_group_assigned.loc[~still_na].copy()
        if len(df_group_unassigned) == 0:
            df_group_unassigned = extra_na.copy()
        else:
            df_group_unassigned = pd.concat([df_group_unassigned, extra_na.drop(columns=["_orig_idx"], errors="ignore")], ignore_index=True)

    if "_orig_idx" in df_group_assigned.columns:
        df_group_assigned = df_group_assigned.drop(columns=["_orig_idx"])

    df_unassigned = pd.concat([df_group_unassigned, df_exceso_cap], ignore_index=True)

    return df_group_assigned.reset_index(drop=True), df_unassigned.reset_index(drop=True)


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

    out["radicacion_viable"] = out["distancia_radicacion_km"].fillna(np.inf) <= 120
    return out


def generar_alertas(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "distancia_centroide" not in out.columns:
        out["distancia_centroide"] = np.nan
    if "distancia_radicacion_km" not in out.columns:
        out["distancia_radicacion_km"] = np.nan
    if "radicacion_viable" not in out.columns:
        out["radicacion_viable"] = False

    if "estatus_asignacion" not in out.columns:
        out["estatus_asignacion"] = None

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

    out.loc[(out["alerta"] == "OK") & (out["estatus_asignacion"] == "Sin radicación viable"), "alerta"] = "SIN RADICACION VIABLE"
    out.loc[(out["alerta"] == "OK") & (out["estatus_asignacion"] == "Sin asignar por capacidad máxima"), "alerta"] = "SIN CAPACIDAD"
    out.loc[(out["alerta"] == "OK") & (out["distancia_radicacion_km"] > 70), "alerta"] = "LEJOS DE RADICACION"
    out.loc[(out["alerta"] == "OK") & (out["distancia_radicacion_km"] > 95), "alerta"] = "ASIGNACION CRITICA"
    out.loc[(out["alerta"] == "OK") & (out["distancia_radicacion_km"] > 120), "alerta"] = "ASIGNACION CRITICA"
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
            f"<b>Semana:</b> {getattr(row, 'SEMANA_RECORRIDO', None)}<br>"
            f"<b>Orden semana:</b> {getattr(row, 'ORDEN_SEMANA', None)}<br>"
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
    "Asignación con radicación dura, compactación territorial, semanas de recorrido y exportación a Google Earth Pro."
)

uploaded = st.file_uploader("Sube tu archivo Excel", type=["xlsx", "xlsm", "xls"])

if uploaded is None:
    st.info(
        "La aplicación detecta columnas geográficas, genera una planeación compacta, permite redistribuir puntos en el mapa, asignar semanas y exportar a Google Earth Pro."
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
min_per = st.sidebar.number_input("Mínimo por entrevistador (meta suave)", min_value=0, value=80, step=1)
max_per = st.sidebar.number_input("Máximo por entrevistador (límite duro)", min_value=1, value=115, step=1)

st.sidebar.subheader("Restricciones geográficas")
hard_max_km = st.sidebar.number_input("Distancia máxima viable a radicación (km)", min_value=10.0, value=120.0, step=5.0)
slack_nearest_km = st.sidebar.number_input("Holgura respecto a la base más cercana (km)", min_value=0.0, value=25.0, step=5.0)
cross_region_max_km = st.sidebar.number_input("Máximo permitido si cambia de región (km)", min_value=0.0, value=45.0, step=5.0)

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

        if len(validos) == 0:
            st.error("No hay registros válidos con coordenadas.")
            st.stop()

        asignados, no_asignados = compact_balanced_assignment(
            validos,
            plantilla_edit,
            lat_col=lat_col,
            lon_col=lon_col,
            min_per=int(min_per),
            max_per=int(max_per),
            priority_col=priority_col,
            hard_max_km=float(hard_max_km),
            slack_nearest_km=float(slack_nearest_km),
            cross_region_max_km=float(cross_region_max_km),
        )

        asignados = recalcular_metricas_asignacion(asignados, plantilla_edit, lat_col, lon_col)
        asignados = generar_alertas(asignados)

        piezas = [asignados]

        if len(no_asignados) > 0:
            if "estatus_asignacion" not in no_asignados.columns:
                no_asignados = preparar_sin_asignar(no_asignados, "Sin asignar")
            else:
                no_asignados = preparar_sin_asignar(no_asignados, "Sin asignar")
                # restaurar estatus originales si existían
            no_asignados = generar_alertas(no_asignados)
            piezas.append(no_asignados)

        if len(invalidos) > 0:
            invalidos = preparar_sin_asignar(invalidos, "Coordenada inválida o a revisar")
            invalidos = generar_alertas(invalidos)
            piezas.append(invalidos)

        resultado = pd.concat(piezas, ignore_index=True)
        resultado = drop_week_columns(resultado)
        resultado = generar_alertas(resultado)

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
resumen_semanal = build_week_summary(resultado)

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
                nuevo_resultado = drop_week_columns(nuevo_resultado)
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

            asignados, no_asignados = compact_balanced_assignment(
                validos,
                plantilla_vigente,
                lat_col=lat_col,
                lon_col=lon_col,
                min_per=int(min_per),
                max_per=int(max_per),
                priority_col=priority_col,
                hard_max_km=float(hard_max_km),
                slack_nearest_km=float(slack_nearest_km),
                cross_region_max_km=float(cross_region_max_km),
            )

            asignados = recalcular_metricas_asignacion(asignados, plantilla_vigente, lat_col, lon_col)
            asignados = generar_alertas(asignados)

            piezas = [asignados]

            if len(no_asignados) > 0:
                no_asignados = preparar_sin_asignar(no_asignados, "Sin asignar")
                no_asignados = generar_alertas(no_asignados)
                piezas.append(no_asignados)

            if len(invalidos) > 0:
                invalidos = preparar_sin_asignar(invalidos, "Coordenada inválida o a revisar")
                invalidos = generar_alertas(invalidos)
                piezas.append(invalidos)

            resultado_reset = pd.concat(piezas, ignore_index=True)
            resultado_reset = drop_week_columns(resultado_reset)
            st.session_state["resultado_planeacion"] = resultado_reset
            st.success("Planeación restablecida.")
            st.rerun()
        except Exception as e:
            st.error(f"No se pudo restablecer la planeación: {e}")

st.markdown("---")
st.subheader("Asignación de semanas de recorrido")

wk1, wk2, wk3 = st.columns([1, 1, 1.4])

num_weeks = wk1.number_input(
    "Número de semanas",
    min_value=1,
    value=4,
    step=1,
    key="num_weeks_planeacion",
)

solo_asignados_actuales = wk2.checkbox(
    "Usar planeación final actual",
    value=True,
    key="solo_asignados_actuales",
)

if wk3.button("Generar semanas de recorrido", type="primary"):
    try:
        base_df = st.session_state["resultado_planeacion"].copy() if solo_asignados_actuales else resultado.copy()
        base_df = drop_week_columns(base_df)

        resultado_semanas = assign_route_weeks(
            df_resultado=base_df,
            plantilla=plantilla_vigente,
            lat_col=lat_col,
            lon_col=lon_col,
            n_weeks=int(num_weeks),
        )

        st.session_state["resultado_planeacion"] = resultado_semanas
        st.success("Semanas de recorrido generadas correctamente.")
        st.rerun()
    except Exception as e:
        st.error(f"No se pudieron generar las semanas: {e}")

resultado = st.session_state["resultado_planeacion"].copy()
resumen = build_summary(resultado, plantilla_vigente)
resumen_semanal = build_week_summary(resultado)

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

tab1, tab2, tab3, tab4 = st.tabs(["Resumen operativo", "Resumen semanal", "Alertas", "Detalle de registros"])

with tab1:
    st.dataframe(resumen, use_container_width=True)

with tab2:
    if resumen_semanal.empty:
        st.info("Aún no se han asignado semanas de recorrido.")
    else:
        st.dataframe(resumen_semanal, use_container_width=True)

with tab3:
    inconsistencias = resultado[resultado["alerta"] != "OK"].copy() if "alerta" in resultado.columns else pd.DataFrame()
    st.dataframe(inconsistencias, use_container_width=True)

with tab4:
    st.dataframe(resultado, use_container_width=True, height=520)

export_bytes = dataframe_to_excel_bytes(
    {
        "asignacion": resultado,
        "resumen_entrevistador": resumen,
        "resumen_semanal": resumen_semanal,
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
                    "min_por_entrevistador_meta_suave",
                    "max_por_entrevistador_limite_duro",
                    "dist_max_radicacion_km",
                    "holgura_base_mas_cercana_km",
                    "max_si_cambia_region_km",
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
                    hard_max_km,
                    slack_nearest_km,
                    cross_region_max_km,
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
st.subheader("Descarga para Google Earth Pro")

default_popup_fields = []
for c in [
    id_col,
    "asignado_a",
    "SUPERV",
    "ROE",
    "MUNICIPIO_RADICACION",
    "SEMANA_RECORRIDO",
    "ORDEN_SEMANA",
    "distancia_radicacion_km",
    "alerta",
    priority_col,
]:
    if c is not None and c in resultado.columns and c not in default_popup_fields:
        default_popup_fields.append(c)

extra_candidates = [c for c in df_raw.columns if c not in default_popup_fields][:6]
default_popup_fields.extend(extra_candidates)

popup_fields = st.multiselect(
    "Variables a mostrar al seleccionar un punto en Google Earth Pro",
    options=list(resultado.columns),
    default=[c for c in default_popup_fields if c in resultado.columns],
    key="popup_fields_google_earth",
)

kmz_bytes = dataframe_to_kmz_bytes(
    df=resultado,
    lat_col=lat_col,
    lon_col=lon_col,
    id_col=id_col,
    popup_fields=popup_fields,
)

st.download_button(
    "Descargar planeación en KMZ para Google Earth Pro",
    data=kmz_bytes,
    file_name="planeacion_operativa_google_earth.kmz",
    mime="application/vnd.google-earth.kmz",
)

st.markdown("---")
st.markdown(
    """
    **Funcionalidades incluidas en esta versión**
    - Planeación automática con radicación dura y compactación territorial.
    - Reasignación manual en mapa por figura.
    - Asignación de semanas de recorrido sobre la planeación final.
    - Resumen semanal por entrevistador.
    - Exportación a Excel.
    - Exportación a KMZ para Google Earth Pro con colores por entrevistador y popup configurable.
    """
)
