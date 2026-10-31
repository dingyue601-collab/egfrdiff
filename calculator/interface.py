"""Streamlit interface for the fitted TabM screening model."""
from html import escape
from pathlib import Path
import logging
import streamlit as st
from clinical_inputs import FIELDS, check_inputs, normalize

DIRECTORY = Path(__file__).resolve().parent


def render_result(probability, threshold, selected, language="en"):
    tr = lambda en, zh: en if language == "en" else zh
    badge = tr("Meets the screening threshold", "达到补测分流阈值，建议加测胱抑素C") if selected else tr("Below the screening threshold", "低于补测分流阈值")
    label = tr("Estimated probability of eGFRdiff &lt; −30%", "eGFR不一致 &lt; −30% 的预测概率")
    threshold_label = tr("Screening threshold", "补测分流阈值")
    st.html(f'<div class="period-label">{label}</div>'
            f'<div class="risk-number">{100 * probability:.2f}%</div>'
            f'<div class="risk-meter"><div class="risk-fill" style="width:{100 * probability:.6f}%"></div>'
            f'<div class="threshold-marker" style="left:{100 * threshold:.6f}%"></div></div>'
            '<div class="scale-labels"><span>0%</span><span>100%</span></div>'
            f'<p class="threshold-caption">{threshold_label} <strong>{100 * threshold:.2f}%</strong></p>'
            f'<div class="risk-badge {"risk-high" if selected else "risk-low"}">{badge}</div>')


@st.cache_resource(show_spinner=False)
def load_model():
    from predictor import Predictor
    return Predictor(DIRECTORY, device="cpu")


def main(language="en"):
    en = language == "en"
    tr = lambda english, chinese: english if en else chinese
    title = tr("eGFRdiff screening calculator", "eGFR不一致识别计算器")
    st.set_page_config(page_title=title, layout="wide")
    for name in ("base.css", "streamlit.css"):
        st.html(DIRECTORY / "assets" / name)
    if not en:
        st.html(DIRECTORY / "assets" / "chinese.css")
    for key in FIELDS:
        st.session_state.setdefault(key, None)
    for key, value in {"scr_unit": "umol", "hb_unit": "gL", "attempted": False,
                       "result": None, "pending": None, "calculation_error": None}.items():
        st.session_state.setdefault(key, value)

    def reset(example=False):
        for key, field in FIELDS.items():
            st.session_state[key] = field[6] if example else None
        if example:
            st.session_state["scr_unit"], st.session_state["hb_unit"] = "umol", "gL"
        for key in ("result", "pending", "calculation_error"):
            st.session_state[key] = None
        st.session_state["attempted"] = False

    def current():
        return normalize(st.session_state, st.session_state["scr_unit"], st.session_state["hb_unit"])

    def calculate(values):
        st.session_state["result"] = None
        st.session_state["calculation_error"] = None
        try:
            prediction = load_model().predict_proba(values).iloc[0].to_dict()
            st.session_state["result"] = {"prediction": prediction, "snapshot": values.copy()}
        except Exception:
            logging.exception("TabM inference failed")
            st.session_state["calculation_error"] = tr(
                "Calculation failed. Check the inputs and model installation, then try again.",
                "计算失败。请核对输入和模型安装后重试。")

    st.markdown(f'<header class="hero"><div class="hero-topline">{tr("TabM · eGFR discordance", "TabM · eGFR不一致")}</div>'
                f'<h1>{title}</h1><p>{tr("Estimate the probability of eGFRdiff < −30% from nine clinical variables.", "根据九项指标，估计 eGFR不一致 < −30% 的概率。")}</p></header>', unsafe_allow_html=True)
    left, right = st.columns([420, 840], gap="large")
    with left, st.container(key="input-card"):
        st.markdown(f'<h2 class="section-title">{tr("Clinical information", "输入指标")}</h2>', unsafe_allow_html=True)
        note = tr("Use laboratory measurements obtained within 1 day of both paired creatinine and cystatin C tests. HbA1c may be measured during the same admission.",
                  "检测指标应在配对肌酐和胱抑素C检测各自前后1天内获得；糖化血红蛋白可采用同次住院结果。")
        st.html(f'<p class="field-note">{note}</p>')
        a, b = st.columns(2)
        a.button(tr("Load example", "载入示例"), on_click=reset, args=(True,), key="load_example")
        b.button(tr("Clear", "清空"), on_click=reset, key="clear")

        def feedback(key):
            bad, soft = check_inputs(current(), language)
            value = current()[key]
            visible = key in soft or key in bad and (st.session_state["attempted"] or value is not None)
            if visible:
                st.markdown(f'<style>.st-key-field_{key} input,.st-key-field_{key} [data-baseweb="select"]>div'
                            '{color:#c62828!important;border-color:#c62828!important;background:#fff8f7!important}</style>'
                            f'<div class="field-feedback">{escape(bad.get(key, soft.get(key, "")))}</div>', unsafe_allow_html=True)

        def number(key):
            field = FIELDS[key]
            label = field[0 if en else 1]
            unit = "" if key in ("scr", "hemoglobin") else field[2 if en else 3]
            with st.container(key="field_" + key):
                st.number_input(label + (" · " + unit if unit else ""), value=None,
                                step=0.1 if key in ("hba1c", "albumin", "ada") else 1.0,
                                format="%g", key=key)
                feedback(key)

        st.markdown(f'<h3 class="group-title">{tr("Patient information", "基本信息")}</h3>', unsafe_allow_html=True)
        number("age")
        with st.container(key="field_sex"):
            st.selectbox(tr("Sex", "性别"), options=(None, 0, 1), key="sex",
                         placeholder=None if en else "请选择性别",
                         format_func=lambda x: tr("Select", "请选择") if x is None else tr("Female", "女性") if x == 0 else tr("Male", "男性"))
            feedback("sex")
        st.markdown(f'<h3 class="group-title">{tr("Kidney function and metabolism", "肾功能与代谢")}</h3>', unsafe_allow_html=True)
        number("scr")
        st.selectbox(tr("Creatinine unit", "肌酐单位"), ("umol", "mg"), key="scr_unit",
                     format_func=lambda x: "μmol/L" if x == "umol" else "mg/dL")
        number("hba1c")
        number("uric_acid")
        st.markdown(f'<h3 class="group-title">{tr("Hematology and biochemistry", "血液与生化")}</h3>', unsafe_allow_html=True)
        number("hemoglobin")
        st.selectbox(tr("Hemoglobin unit", "血红蛋白单位"), ("gL", "gdL"), key="hb_unit",
                     format_func=lambda x: "g/L" if x == "gL" else "g/dL")
        for key in ("albumin", "alp", "ada"):
            number(key)
        pressed = st.button(tr("Calculate probability", "计算概率") + " →", key="calculate", type="primary", width="stretch")
        if pressed:
            st.session_state["attempted"] = True
            st.session_state["calculation_error"] = None
            st.session_state["pending"] = None
            bad, soft = check_inputs(current(), language)
            if bad:
                st.session_state["result"] = None
                st.rerun()
            elif soft:
                st.session_state["pending"] = current()
            else:
                calculate(current())

    if st.session_state["pending"] is not None:
        @st.dialog(tr("Check measurements", "请核对输入"), dismissible=False)
        def confirm():
            pending = st.session_state["pending"]
            _, soft = check_inputs(pending, language)
            for message in soft.values():
                st.write(message)
            a, b = st.columns(2)
            if a.button(tr("Edit", "返回修改"), key="edit_values"):
                st.session_state["pending"] = None
                st.rerun()
            if b.button(tr("Confirm and calculate", "数值无误，继续计算"), key="confirm_values", type="primary"):
                st.session_state["pending"] = None
                if pending == current():
                    calculate(pending)
                else:
                    st.session_state["result"] = None
                    st.session_state["calculation_error"] = tr("Inputs changed. Check them and calculate again.", "输入已更改，请核对后重新计算。")
                st.rerun()
        confirm()

    result = st.session_state["result"]
    fresh = result is not None and result["snapshot"] == current()
    with right:
        with st.container(key="risk-card"):
            st.markdown(f'<div class="result-heading"><span class="eyebrow">{tr("Result", "计算结果")}</span></div>', unsafe_allow_html=True)
            if st.session_state["calculation_error"]:
                st.error(st.session_state["calculation_error"])
            elif st.session_state["attempted"] and check_inputs(current(), language)[0]:
                st.markdown(f'<div class="empty-result"><h2>{tr("Check inputs", "请核对输入")}</h2><p>{tr("Complete the fields marked in red.", "请补全标红指标，并核对数值及单位。")}</p></div>', unsafe_allow_html=True)
            elif result is None:
                st.markdown(f'<div class="empty-result"><div class="empty-symbol">—</div><h2>{tr("Ready to calculate", "等待计算")}</h2><p>{tr("Enter the clinical variables and calculate the probability.", "填写指标后，点击计算概率。")}</p></div>', unsafe_allow_html=True)
            elif not fresh:
                st.warning(tr("Inputs changed. Recalculate to update the result.", "输入已更改，请重新计算。"))
                st.markdown(f'<div class="empty-result"><h2>{tr("Recalculation required", "请重新计算")}</h2></div>', unsafe_allow_html=True)
            else:
                prediction = result["prediction"]
                probability, threshold = prediction["probability_negative_discordance"], prediction["threshold"]
                selected = prediction["screen_positive"]
                render_result(probability, threshold, selected, language)
        with st.container(key="summary-card"):
            st.markdown(f'<h2 class="section-title">{tr("Input summary", "本次输入")}</h2>', unsafe_allow_html=True)
            if fresh:
                rows = []
                for key, field in FIELDS.items():
                    value = result["snapshot"][key]
                    display = tr("Male", "男性") if key == "sex" and value == 1 else tr("Female", "女性") if key == "sex" else f"{value:g} {field[2 if en else 3]}"
                    rows.append(f"<tr><th>{escape(field[0 if en else 1])}</th><td>{escape(display)}</td></tr>")
                st.markdown(f'<table class="input-table"><thead><tr><th>{tr("Variable", "指标")}</th><th>{tr("Value", "数值")}</th></tr></thead><tbody>{"".join(rows)}</tbody></table>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="summary-empty">{tr("Inputs will appear here after calculation.", "计算后显示本次输入。")}</p>', unsafe_allow_html=True)
        with st.container(key="model-card"):
            st.markdown(f'<h2 class="section-title">{tr("About the model", "模型说明")}</h2>'
                        '<div class="model-info">'
                        f'<p><strong>{tr("Target:", "识别目标：")}</strong> {tr("eGFRdiff = 100 × (eGFRcys − eGFRcr) / eGFRcr &lt; −30%.", "eGFR不一致 = 100 ×（胱抑素C计算的eGFR − 肌酐计算的eGFR）/ 肌酐计算的eGFR &lt; −30%。")}</p>'
                        f'<p><strong>{tr("Population:", "适用人群：")}</strong> {tr("Adults hospitalized with type 2 diabetic kidney disease.", "糖尿病肾病成年住院患者。")}</p>'
                        f'<p><strong>{tr("Model version:", "模型版本：")}</strong> <a href="https://github.com/yandex-research/tabm" target="_blank" rel="noopener noreferrer">TabM 0.0.3</a></p></div>', unsafe_allow_html=True)
    with st.container(key="batch-card"):
        from batch_interface import show_batch
        show_batch(load_model, render_result, language)
    st.markdown(f'<footer>{tr("Supports targeted cystatin C testing.", "辅助选择胱抑素C加测对象。")}</footer>', unsafe_allow_html=True)
