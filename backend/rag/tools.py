from __future__ import annotations

import json
import math
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass


@dataclass
class ToolResult:
    name: str
    text: str


WEATHER_RE = re.compile(
    r"\b(meteo|prognoz|vreme|temperatur|vânt|ploa[ăi]|z[ăa]pad|"
    r"weather|forecast|temperature|wind|rain|snow)",
    re.I,
)
DISTANCE_RE = re.compile(
    r"\b(distan[țt][ăa]|distanta|cât de departe|km|kilometri|cât face|"
    r"distance|how far|far from)",
    re.I,
)


def detect_intent(query: str) -> str | None:
    if WEATHER_RE.search(query):
        return "weather"
    if DISTANCE_RE.search(query):
        return "distance"
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def fetch_weather(lat: float, lon: float, *, timeout: float = 10.0) -> ToolResult:
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,weather_code,wind_speed_10m,precipitation"
        "&timezone=auto"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)
    c = data.get("current", {})
    temp = c.get("temperature_2m")
    wind = c.get("wind_speed_10m")
    precip = c.get("precipitation", 0)
    text = (
        f"Prognoză curentă Open-Meteo pentru coordonatele {lat:.3f}, {lon:.3f}: "
        f"temperatură {temp}°C, vânt {wind} km/h, precipitații {precip} mm."
    )
    return ToolResult(name="weather", text=text)


def compute_distance(
    lat1: float, lon1: float, lat2: float, lon2: float,
    name1: str | None = None, name2: str | None = None,
) -> ToolResult:
    km = haversine_km(lat1, lon1, lat2, lon2)
    a = f"{name1} ({lat1:.3f}, {lon1:.3f})" if name1 else f"({lat1:.3f}, {lon1:.3f})"
    b = f"{name2} ({lat2:.3f}, {lon2:.3f})" if name2 else f"({lat2:.3f}, {lon2:.3f})"
    text = f"Distanța în linie dreaptă între {a} și {b} este {km:.1f} km."
    return ToolResult(name="distance", text=text)
