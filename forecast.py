from pathlib import Path

import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingRegressor

DATA_DIR = Path(__file__).parent / "data"
LOAD_FILE = DATA_DIR / "AEP_hourly.csv"
WEATHER_FILE = DATA_DIR / "weather.csv"

FEATURES = ["hour", "dayofweek", "lag_24", "lag_168",
            "temp", "temp_lag_24", "temp_diff_24", "temp_roll_24"]
SPLIT_DATE = "2017-01-01"


def load_consumption():
    """Читает потребление и приводит ряд к ровной почасовой сетке."""
    df = pd.read_csv(LOAD_FILE, parse_dates=["Datetime"])
    series = df.groupby("Datetime")["AEP_MW"].mean()
    full_range = pd.date_range(series.index.min(), series.index.max(), freq="h")
    return series.reindex(full_range).interpolate().rename("load")


def load_weather(index):
    """Берёт температуру из файла-кэша или скачивает с Open-Meteo."""
    if WEATHER_FILE.exists():
        temp = pd.read_csv(WEATHER_FILE, index_col=0, parse_dates=True)["temp"]
    else:
        frames = []
        for year in range(index.min().year, index.max().year + 1):
            params = {
                "latitude": 39.96,
                "longitude": -82.99,
                "start_date": f"{year}-01-01",
                "end_date": f"{year}-12-31",
                "hourly": "temperature_2m",
                "timezone": "America/New_York",
            }
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            frames.append(pd.DataFrame({"Datetime": pd.to_datetime(h["time"]),
                                        "temp": h["temperature_2m"]}))
        temp = pd.concat(frames).drop_duplicates("Datetime").set_index("Datetime")["temp"]
        temp.to_csv(WEATHER_FILE)
    return temp.reindex(index).interpolate()


def build_features(load, temp):
    """Собирает таблицу признаков для модели."""
    d = pd.DataFrame({"load": load, "temp": temp})
    d["hour"] = d.index.hour
    d["dayofweek"] = d.index.dayofweek
    d["lag_24"] = d["load"].shift(24)
    d["lag_168"] = d["load"].shift(168)
    d["temp_lag_24"] = d["temp"].shift(24)
    d["temp_diff_24"] = d["temp"] - d["temp_lag_24"]
    d["temp_roll_24"] = d["temp"].rolling(24).mean()
    return d.dropna()


def metrics(actual, predicted):
    """Возвращает среднюю ошибку в MW (MAE) и в процентах (MAPE)."""
    error = (actual - predicted).abs()
    return error.mean(), (error / actual).mean() * 100


def get_results():
    """Обучает модель на данных до 2017 года и возвращает прогнозы на проверочный период."""
    load = load_consumption()
    temp = load_weather(load.index)
    data = build_features(load, temp)

    train = data.loc[:"2016-12-31"]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, random_state=42)
    model.fit(train[FEATURES], train["load"])

    test = data.loc[SPLIT_DATE:].copy()
    test["model"] = model.predict(test[FEATURES])
    test["baseline"] = test["lag_168"]
    return test[["load", "model", "baseline"]]