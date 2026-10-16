import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from forecast import get_results, metrics

st.set_page_config(page_title="Прогноз энергопотребления", layout="wide")


@st.cache_data(show_spinner="Загружаю данные и обучаю модель...")
def load_results():
    return get_results()


results = load_results()

st.title("Прогноз почасового энергопотребления")
st.caption("Энергосеть AEP (США). Прогноз на сутки вперёд, проверка на данных 2017–2018 годов.")

mae_all, mape_all = metrics(results["load"], results["model"])
_, mape_base = metrics(results["load"], results["baseline"])

c1, c2, c3 = st.columns(3)
c1.metric("Ошибка модели (MAPE)", f"{mape_all:.2f} %")
c2.metric("Ошибка модели (MAE)", f"{mae_all:.0f} MW")
c3.metric("Ошибка базовой модели (MAPE)", f"{mape_base:.2f} %")

st.sidebar.header("Что показать")
day = st.sidebar.date_input(
    "Начальная дата",
    value=pd.Timestamp("2017-07-05").date(),
    min_value=results.index.min().date(),
    max_value=results.index.max().date(),
)
days = st.sidebar.slider("Сколько дней показать", 1, 14, 3)

start = pd.Timestamp(day)
end = start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
window = results.loc[start:end]

st.subheader(f"Потребление, MW: {start:%d.%m.%Y} + {days} дн.")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(window.index, window["load"], label="Факт", linewidth=2)
ax.plot(window.index, window["model"], label="Модель")
ax.plot(window.index, window["baseline"], label="Базовая", alpha=0.7)
ax.set_ylabel("MW")
ax.legend()
ax.grid(alpha=0.3)
fig.autofmt_xdate()
st.pyplot(fig)

mae_w, mape_w = metrics(window["load"], window["model"])
mae_bw, mape_bw = metrics(window["load"], window["baseline"])

col1, col2 = st.columns(2)
col1.metric("Модель на выбранном периоде", f"{mape_w:.2f} %")
col1.caption(f"MAE {mae_w:.0f} MW")
col2.metric("Базовая на выбранном периоде", f"{mape_bw:.2f} %")
col2.caption(f"MAE {mae_bw:.0f} MW")

with st.expander("10 дней с самой большой ошибкой модели"):
    err = (results["load"] - results["model"]).abs()
    worst = err.groupby(err.index.date).mean().sort_values(ascending=False).head(10)
    st.dataframe(worst.rename("Средняя ошибка, MW").round(0))