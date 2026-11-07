"""
Tool del agente que invoca el pipeline de ML de regresión de precios de casas.

El pipeline (Modelo-ML/precio_casas_pipeline.joblib) necesita 37 columnas del
dataset House Prices de Ames. En una conversación el usuario solo dará unas
pocas, así que la tool expone las más influyentes como parámetros opcionales y
completa el resto con valores típicos del train.csv (mediana para numéricas,
moda para categóricas). El modelo predice log(SalePrice): se aplica np.exp
para devolver el precio en dólares, igual que en la app de Streamlit.
"""

import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

RUTA_MODELO_ML = Path(__file__).resolve().parent.parent / "Modelo-ML"
sys.path.insert(0, str(RUTA_MODELO_ML))

import preprocessors as pp  # noqa: E402
from configuraciones import config  # noqa: E402

# El .joblib se entrenó con `import preprocessors as pp` (top-level), así que
# joblib.load necesita resolver `preprocessors` sin prefijo.
sys.modules.setdefault("preprocessors", pp)

RUTA_PIPELINE = RUTA_MODELO_ML / "precio_casas_pipeline.joblib"

# Valores típicos del train.csv para las columnas que el usuario no mencione.
VALORES_POR_DEFECTO = {
    "MSSubClass": 20,
    "MSZoning": "RL",
    "LotFrontage": 69.0,
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "LotConfig": "Inside",
    "Neighborhood": "NAmes",
    "OverallQual": 6,
    "OverallCond": 5,
    "YearRemodAdd": 1994,
    "RoofStyle": "Gable",
    "Exterior1st": "VinylSd",
    "ExterQual": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "Unf",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "1stFlrSF": 1087,
    "2ndFlrSF": 0,
    "GrLivArea": 1464,
    "BsmtFullBath": 0,
    "HalfBath": 0,
    "KitchenQual": "TA",
    "TotRmsAbvGrd": 6,
    "Functional": "Typ",
    "Fireplaces": 0,
    "FireplaceQu": None,
    "GarageFinish": "Unf",
    "GarageCars": 2,
    "GarageArea": 480,
    "PavedDrive": "Y",
    "WoodDeckSF": 0,
    "ScreenPorch": 0,
    "SaleCondition": "Normal",
    "YrSold": 2008,
}

Calidad = Literal["Ex", "Gd", "TA", "Fa", "Po"]
Barrio = Literal[
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NoRidge",
    "NPkVill", "NridgHt", "NWAmes", "OldTown", "SWISU", "Sawyer", "SawyerW",
    "Somerst", "StoneBr", "Timber", "Veenker",
]


class EntradaPrediccion(BaseModel):
    """Características de la casa. Solo area_habitable_pies2 es obligatoria."""

    area_habitable_pies2: float = Field(
        description="Área habitable sobre el nivel del suelo en pies cuadrados (GrLivArea). 1 m² = 10.7639 ft².",
        gt=0,
    )
    area_primer_piso_pies2: Optional[float] = Field(None, description="Área del primer piso en ft² (1stFlrSF).")
    area_segundo_piso_pies2: Optional[float] = Field(None, description="Área del segundo piso en ft² (2ndFlrSF). 0 si es de un solo piso.")
    calidad_general: Optional[int] = Field(None, ge=1, le=10, description="Calidad general de materiales y acabados, 1 (muy mala) a 10 (excelente) (OverallQual).")
    condicion_general: Optional[int] = Field(None, ge=1, le=9, description="Estado de conservación general, 1 a 9 (OverallCond).")
    anio_remodelacion: Optional[int] = Field(None, ge=1872, le=2010, description="Año de construcción o de la última remodelación (YearRemodAdd).")
    anio_venta: Optional[int] = Field(None, ge=2006, le=2010, description="Año de venta, entre 2006 y 2010 (YrSold).")
    barrio: Optional[Barrio] = Field(None, description="Barrio de Ames, Iowa (Neighborhood). NoRidge, NridgHt y StoneBr son los más caros.")
    total_habitaciones: Optional[int] = Field(None, ge=1, le=20, description="Total de ambientes sobre el nivel del suelo, sin contar baños (TotRmsAbvGrd).")
    medios_banos: Optional[int] = Field(None, ge=0, le=3, description="Medios baños sobre el nivel del suelo (HalfBath).")
    banos_completos_sotano: Optional[int] = Field(None, ge=0, le=3, description="Baños completos en el sótano (BsmtFullBath).")
    autos_garaje: Optional[int] = Field(None, ge=0, le=4, description="Capacidad del garaje en autos (GarageCars). 0 si no tiene garaje.")
    area_garaje_pies2: Optional[float] = Field(None, ge=0, description="Área del garaje en ft² (GarageArea).")
    chimeneas: Optional[int] = Field(None, ge=0, le=3, description="Número de chimeneas (Fireplaces).")
    calidad_exterior: Optional[Calidad] = Field(None, description="Calidad del exterior: Ex excelente, Gd buena, TA normal, Fa regular, Po mala (ExterQual).")
    calidad_cocina: Optional[Calidad] = Field(None, description="Calidad de la cocina, misma escala (KitchenQual).")
    calidad_sotano: Optional[Calidad] = Field(None, description="Calidad (altura) del sótano, misma escala (BsmtQual).")
    aire_acondicionado_central: Optional[bool] = Field(None, description="True si tiene aire acondicionado central (CentralAir).")
    frente_lote_pies: Optional[float] = Field(None, gt=0, description="Pies lineales de calle conectados al lote (LotFrontage).")


@lru_cache(maxsize=1)
def cargar_pipeline():
    """Carga el pipeline una sola vez por proceso."""
    return joblib.load(RUTA_PIPELINE)


def construir_registro(entrada: EntradaPrediccion) -> dict:
    """Mezcla lo que dio el usuario con los valores por defecto, manteniendo coherencia."""
    registro = dict(VALORES_POR_DEFECTO)

    mapeo = {
        "calidad_general": "OverallQual",
        "condicion_general": "OverallCond",
        "anio_remodelacion": "YearRemodAdd",
        "anio_venta": "YrSold",
        "barrio": "Neighborhood",
        "total_habitaciones": "TotRmsAbvGrd",
        "medios_banos": "HalfBath",
        "banos_completos_sotano": "BsmtFullBath",
        "autos_garaje": "GarageCars",
        "area_garaje_pies2": "GarageArea",
        "chimeneas": "Fireplaces",
        "calidad_exterior": "ExterQual",
        "calidad_cocina": "KitchenQual",
        "calidad_sotano": "BsmtQual",
        "frente_lote_pies": "LotFrontage",
    }
    for campo, columna in mapeo.items():
        valor = getattr(entrada, campo)
        if valor is not None:
            registro[columna] = valor

    if entrada.aire_acondicionado_central is not None:
        registro["CentralAir"] = "Y" if entrada.aire_acondicionado_central else "N"

    # Áreas: GrLivArea = 1stFlrSF + 2ndFlrSF en el dataset.
    area = entrada.area_habitable_pies2
    segundo = entrada.area_segundo_piso_pies2
    primero = entrada.area_primer_piso_pies2
    if primero is None and segundo is None:
        primero, segundo = area, 0
    elif primero is None:
        primero = max(area - segundo, 0)
    elif segundo is None:
        segundo = max(area - primero, 0)
    registro.update({"GrLivArea": area, "1stFlrSF": primero, "2ndFlrSF": segundo})

    # Sin garaje: el dataset registra área 0 y GarageFinish faltante.
    if registro["GarageCars"] == 0:
        registro["GarageArea"] = 0
        registro["GarageFinish"] = None
    elif entrada.area_garaje_pies2 is None:
        registro["GarageArea"] = 240 * registro["GarageCars"]

    # Con chimenea, la calidad típica es Gd; sin chimenea queda faltante.
    if registro["Fireplaces"] > 0:
        registro["FireplaceQu"] = "Gd"

    return registro


def predecir(registro: dict) -> float:
    """Ejecuta el pipeline sobre un registro y devuelve el precio en dólares."""
    df = pd.DataFrame([registro])
    df[config.CATEGORICAL_VARS] = df[config.CATEGORICAL_VARS].astype(str)
    df = df[config.FEATURES]
    prediccion_log = cargar_pipeline().predict(df)
    return float(np.exp(prediccion_log[0]))


@tool("predecir_precio_casa", args_schema=EntradaPrediccion)
def predecir_precio_casa(**kwargs) -> dict:
    """Estima el precio de venta (USD) de una casa de Ames, Iowa, con el modelo de ML de regresión.

    Solo area_habitable_pies2 es obligatoria; los demás parámetros son opcionales
    y, si no se envían, se completan con valores típicos del dataset.
    Devuelve el precio estimado y las características usadas.
    """
    entrada = EntradaPrediccion(**kwargs)
    registro = construir_registro(entrada)
    precio = predecir(registro)
    return {
        "precio_estimado_usd": round(precio, 2),
        "caracteristicas_indicadas_por_usuario": entrada.model_dump(exclude_none=True),
        "caracteristicas_usadas_por_el_modelo": {
            k: v for k, v in registro.items() if k in config.FEATURES
        },
    }
