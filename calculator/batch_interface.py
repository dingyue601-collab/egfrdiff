"""Batch CSV interface."""
import hashlib
import pandas as pd
import streamlit as st
from batch_io import template, read_csv, check_rows, predict_batch, csv_bytes, column_labels


def show_batch(load_model, render_result, language="en"):
    tr = lambda en, zh: en if language == "en" else zh
    st.subheader(tr("Batch records", "批量记录"))
    st.caption(tr("Use the template units. Patient is a unique record identifier; 123 is a synthetic example. Sex: 0 = Female, 1 = Male.",
                  "按模板单位填写。患者编号为每条记录的唯一编号，可自定义；123为示例记录，性别0为女性、1为男性。"))
    st.download_button(tr("Download CSV template", "下载CSV模板"), csv_bytes(template(language)), "egfrdiff_template.csv", "text/csv")
    upload = st.file_uploader(tr("Upload CSV", "上传CSV"), type=["csv"], key="upload_" + language)
    content = upload.getvalue() if upload else csv_bytes(template())
    key = hashlib.sha256(content).hexdigest()[:12]
    try:
        frame = read_csv(content)
    except (ValueError, UnicodeError) as exc:
        st.error(str(exc))
        return
    labels = column_labels(language)
    columns = {name: st.column_config.TextColumn(labels.get(name, name)) for name in frame.columns}
    columns["Patient"] = st.column_config.TextColumn(labels.get("Patient", "Patient"), required=True)
    for name in frame.columns.drop("Patient"):
        values = pd.to_numeric(frame[name].replace("", None), errors="coerce")
        # Keep invalid text editable until it is corrected.
        if (values.notna() | frame[name].eq("")).all():
            frame[name] = values
            columns[name] = st.column_config.NumberColumn(labels.get(name, name), format="%g")
    edited = st.data_editor(frame, num_rows="dynamic", hide_index=True, width="stretch",
                            column_config=columns, key="batch_" + key)
    _, _, errors, warnings, cells = check_rows(edited, language)
    if errors.ne("").any() or warnings.ne("").any():
        preview = edited.copy()
        preview[tr("Input check", "输入核对")] = errors.where(errors.ne(""), warnings)
        def highlight(row):
            bad = cells[edited.index.get_loc(row.name)]
            return ["color:#c62828;background-color:#fff0ee" if col in bad else "" for col in row.index]
        st.dataframe(preview.style.apply(highlight, axis=1), hide_index=True, width="stretch", column_config=columns)
    confirm = st.checkbox(tr("I have checked measurements outside the displayed ranges", "已核对提示范围外的检测结果"), key="confirm_batch_" + key) if warnings.ne("").any() else False
    snapshot = edited.to_json(orient="split") + str(confirm)
    if st.button(tr("Calculate batch", "批量计算"), type="primary", key="calculate_batch"):
        if not len(edited):
            st.warning(tr("Add at least one record", "请至少添加一条记录"))
        else:
            with st.spinner(tr("Calculating", "正在计算")):
                result = predict_batch(edited, load_model(), language, confirm)
            st.session_state["batch_result"] = (snapshot, result)
    saved = st.session_state.get("batch_result")
    if not saved or saved[0] != snapshot:
        return
    result = saved[1]
    st.dataframe(result, hide_index=True, width="stretch", column_config={
        **columns, "Status": labels.get("Status", "Status"),
        "recommend_testing": labels.get("recommend_testing", "recommend_testing"),
        "p_positive": st.column_config.NumberColumn(tr("Probability", "eGFR不一致 < −30% 的概率"), format="percent"),
        "p_negative": None, "threshold": None})
    st.download_button(tr("Download results CSV", "导出结果CSV"), csv_bytes(result, language), "egfrdiff_results.csv", "text/csv")
    computed = result[result["p_positive"].notna()]
    if len(computed):
        selected = st.selectbox(tr("Select a patient", "选择患者"), computed["Patient"].tolist())
        row = computed[computed["Patient"] == selected].iloc[0]
        with st.container(key="batch-result"):
            st.html(f'<div class="result-heading"><span class="eyebrow">{tr("Result", "计算结果")}</span></div>')
            render_result(row["p_positive"], row["threshold"], row["recommend_testing"], language)
