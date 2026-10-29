"""Демо-приложение сервиса винных подборок: модель отбирает хорошие вина по лабораторному анализу."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn import metrics

APP_DIR = Path(__file__).parent

# st.cache_resource держит один объект модели на весь сервер: он не копируется между сессиями
# и не сериализуется, поэтому подходит для неизменяемых тяжелых объектов.
# st.cache_data наоборот возвращает копию результата, что нужно для таблиц и массивов:
# если пользователь их изменит, кэш не испортится.
# Без декоратора joblib.load вызывался бы при каждом движении слайдера, те модель читалась
# бы с диска заново на каждое действие пользователя.


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / "model.joblib")


@st.cache_data
def load_test():
    return pd.read_csv(APP_DIR / "test.csv")


@st.cache_data
def load_config():
    return json.loads((APP_DIR / "config.json").read_text())


@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test_df = load_test()
config = load_config()

FEATURES = [c for c in test_df.columns if c != "target"]
NUMERIC = [c for c in FEATURES if c != "is_red"]

X_test = test_df[FEATURES]
y_test = test_df["target"].to_numpy()
proba = test_proba(model, X_test)

st.set_page_config(page_title="Подборка вин", layout="wide")
st.title("Отбор хороших вин по лабораторному анализу")

st.sidebar.header("Порог")
threshold = st.sidebar.slider(
    "Порог вероятности", 0.0, 1.0, float(config["threshold"]),
    step=0.001, format="%.3f", key="threshold"
)
st.sidebar.caption(
    f"Вино попадает в подборку, если вероятность не ниже {threshold:.3f}. "
    "Порог общий для всех вкладок."
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Оценка вина", "Модельные метрики", "Пакетное предсказание", "Что выучила модель"]
)

with tab1:
    st.subheader("Оценка одного вина")
    st.write("Введите результаты анализа, приложение покажет вероятность и решение модели.")

    values = {}
    cols = st.columns(3)
    for i, feature in enumerate(NUMERIC):
        column = test_df[feature]
        values[feature] = cols[i % 3].number_input(
            feature,
            min_value=float(column.min()),
            max_value=float(column.max()),
            value=float(column.median()),
            step=float((column.max() - column.min()) / 100),
            format="%.4f",
        )

    color = st.radio("Цвет вина", ["белое", "красное"], horizontal=True)
    values["is_red"] = 1 if color == "красное" else 0

    sample = pd.DataFrame([values])[FEATURES]
    p = float(model.predict_proba(sample)[0, 1])

    left, right = st.columns(2)
    left.metric("Вероятность хорошего вина", f"{p:.3f}")
    right.metric("Решение при пороге " + f"{threshold:.3f}", "в подборку" if p >= threshold else "мимо")
    st.progress(p)

with tab2:
    st.subheader("Метрики на тестовой выборке")
    pred = (proba >= threshold).astype(int)

    precision = metrics.precision_score(y_test, pred, zero_division=0)
    recall = metrics.recall_score(y_test, pred, zero_division=0)
    f1 = metrics.f1_score(y_test, pred, zero_division=0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", f"{precision:.3f}")
    c2.metric("Recall", f"{recall:.3f}")
    c3.metric("F1", f"{f1:.3f}")

    left, right = st.columns([1, 2])

    with left:
        cm = metrics.confusion_matrix(y_test, pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
        metrics.ConfusionMatrixDisplay(
            cm, display_labels=["обычное", "хорошее"]
        ).plot(ax=ax_cm, colorbar=False)
        ax_cm.set_title("Матрица ошибок")
        st.pyplot(fig_cm)

    with right:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.5))

        fpr, tpr, roc_thr = metrics.roc_curve(y_test, proba)
        ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {metrics.roc_auc_score(y_test, proba):.3f}")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray")
        i_roc = int(np.argmin(np.abs(roc_thr - threshold)))
        ax_roc.scatter(fpr[i_roc], tpr[i_roc], color="red", zorder=5, s=70)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_ylabel("TPR")
        ax_roc.set_title("ROC-кривая")
        ax_roc.legend()

        pr, rc, pr_thr = metrics.precision_recall_curve(y_test, proba)
        ax_pr.plot(rc, pr, label=f"PR-AUC = {metrics.average_precision_score(y_test, proba):.3f}")
        ax_pr.axhline(y_test.mean(), ls="--", color="gray")
        ax_pr.scatter(recall, precision, color="red", zorder=5, s=70)
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.set_title("PR-кривая")
        ax_pr.legend()

        st.pyplot(fig)

    st.info(
        f"Сейчас в подборку попадает {pred.sum()} из {len(pred)} вин. "
        f"Из них действительно хороших {precision:.0%}, и модель находит {recall:.0%} "
        "всех хороших вин теста. Если поднять порог, подборка станет меньше и чище: "
        "доля разочарований упадет, но часть хороших вин потеряется. "
        "Если опустить порог, подборка вырастет, хороших вин в ней найдется больше, "
        "но вместе с ними придет больше посредственных."
    )

with tab3:
    st.subheader("Пакетное предсказание")
    st.write("Загрузите CSV с признаками вин без столбца target.")

    uploaded = st.file_uploader("CSV-файл", type="csv")
    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as error:
            st.error(f"Не удалось прочитать файл: {error}")
        else:
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error("В файле не хватает столбцов: " + ", ".join(missing))
            else:
                batch_proba = model.predict_proba(batch[FEATURES])[:, 1]
                result = batch.copy()
                result["probability"] = batch_proba.round(4)
                result["в подборку"] = np.where(batch_proba >= threshold, "да", "нет")

                st.write(f"Вин в файле: {len(result)}, из них в подборку попадает "
                         f"{int((batch_proba >= threshold).sum())}.")
                st.dataframe(result)
                st.download_button(
                    "Скачать результат",
                    result.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )

with tab4:
    st.subheader("Коэффициенты логистической регрессии")
    st.write("Признаки стандартизованы, поэтому коэффициенты сравнимы между собой.")

    coefs = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values()

    fig_coef, ax_coef = plt.subplots(figsize=(8, 6))
    ax_coef.barh(coefs.index, coefs.values,
                 color=np.where(coefs.values > 0, "tab:green", "tab:red"))
    ax_coef.axvline(0, color="black", lw=0.8)
    ax_coef.set_xlabel("вес признака")
    st.pyplot(fig_coef)

    top_pos = ", ".join(coefs.tail(3).index[::-1])
    top_neg = ", ".join(coefs.head(3).index)
    st.write(f"Сильнее всего за хорошее вино голосуют: {top_pos}.")
    st.write(f"Против хорошего вина работают: {top_neg}.")
