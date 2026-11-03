"""CSV input and batch inference for the nine-variable model."""
import codecs
import csv
from io import StringIO
import pandas as pd
from clinical_inputs import FIELDS, check_inputs

HEADERS = {
    "age": "Age (years)", "sex": "Sex (0=Female, 1=Male)",
    "scr": "Serum creatinine (umol/L)", "hba1c": "HbA1c (%)",
    "hemoglobin": "Hemoglobin (g/L)", "uric_acid": "Serum uric acid (umol/L)",
    "albumin": "Serum albumin (g/L)", "alp": "Alkaline phosphatase (U/L)",
    "ada": "Adenosine deaminase (U/L)",
}


def column_labels(language="en"):
    if language == "en":
        return {}
    names = {HEADERS[k]: f"{v[1]}（{v[3]}）" if v[3] else v[1] for k, v in FIELDS.items()}
    names[HEADERS["sex"]] = "性别（0=女性，1=男性）"
    names.update({"Patient": "患者编号", "Status": "计算状态", "p_positive": "eGFR不一致 < −30% 的概率",
                  "p_negative": "非负向不一致概率", "threshold": "补测分流阈值", "recommend_testing": "建议加测胱抑素C"})
    return names


def template(language="en"):
    frame = pd.DataFrame([{"Patient": "123", **{HEADERS[k]: v[6] for k, v in FIELDS.items()}}])
    return frame.rename(columns=column_labels(language))


def read_csv(content):
    """Read UTF-8, GB18030 or BOM-marked UTF-16 without repairing malformed rows."""
    if content.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = content.decode("utf-16")
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = content.decode("gb18030")
            except UnicodeDecodeError as exc:
                raise ValueError("Unsupported CSV encoding / 请使用UTF-8、GB18030或UTF-16编码") from exc
    if "\x00" in text:
        raise ValueError("CSV contains null characters / CSV包含无效空字符")
    reader = csv.reader(StringIO(text), strict=True)
    try:
        headers = [name.strip() for name in next((row for row in reader if row), [])]
        rows = []
        for row in reader:
            if not row:
                continue
            if len(row) != len(headers):
                raise ValueError(f"CSV line {reader.line_num}: wrong number of fields / 第{reader.line_num}行列数与表头不一致")
            rows.append(row)
    except csv.Error as exc:
        raise ValueError(f"Invalid CSV near line {reader.line_num} / 第{reader.line_num}行附近CSV格式错误") from exc
    aliases = {**HEADERS, **{v: k for k, v in column_labels("zh").items()}}
    columns = [aliases.get(name, name) for name in headers]
    if len(set(headers)) != len(headers) or len(set(columns)) != len(columns):
        raise ValueError("Duplicate column names / 列名重复")
    frame = pd.DataFrame(rows, columns=columns)
    missing = [name for name in HEADERS.values() if name not in frame]
    if "Patient" not in frame or missing:
        raise ValueError("Use the CSV template / 请使用CSV模板: " + ", ".join(missing))
    return frame[["Patient", *HEADERS.values()]]


def check_rows(frame, language="en"):
    numeric = frame.rename(columns={v: k for k, v in HEADERS.items()})[list(FIELDS)].apply(pd.to_numeric, errors="coerce")
    ids = frame["Patient"].fillna("").astype(str).str.strip()
    duplicate = ids.duplicated(keep=False)
    problems, warnings, cells = [], [], []
    for i, (_, row) in enumerate(numeric.iterrows()):
        bad, soft = check_inputs(row.to_dict(), language)
        if not ids.iloc[i] or duplicate.iloc[i]:
            bad["Patient"] = "Missing or duplicate record ID" if language == "en" else "记录编号缺失或重复"
        names = column_labels(language)
        label = lambda key: names.get("Patient", "Patient") if key == "Patient" else names.get(HEADERS[key], HEADERS[key])
        problems.append("; ".join(f"{label(k)}: {v}" for k, v in bad.items()))
        warnings.append("; ".join(f"{label(k)}: {v}" for k, v in soft.items()))
        cells.append({"Patient" if k == "Patient" else HEADERS[k] for k in bad.keys() | soft.keys()})
    return numeric, ids, pd.Series(problems, index=frame.index), pd.Series(warnings, index=frame.index), cells


def predict_batch(frame, predictor, language="en", confirm_ranges=False):
    frame = frame.reset_index(drop=True)
    numeric, ids, errors, warnings, _ = check_rows(frame, language)
    valid = errors.eq("") & (warnings.eq("") | confirm_ranges)
    result = frame.copy()
    result["Patient"] = ids
    result["Status"] = errors.where(errors.ne(""), warnings.where(~valid, "Ready"))
    for name in ("p_positive", "p_negative", "threshold"):
        result[name] = float("nan")
    result["recommend_testing"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    if valid.any():
        prediction = predictor.predict_proba(numeric.loc[valid]).reset_index(drop=True)
        result.loc[valid, ["p_positive", "p_negative", "threshold"]] = prediction[["p_positive", "p_negative", "threshold"]].to_numpy()
        result.loc[valid, "recommend_testing"] = prediction["recommend_testing"].to_numpy(dtype=bool)
        result.loc[valid, "Status"] = "Calculated" if language == "en" else "已计算"
    return result


def csv_bytes(frame, language="en"):
    export = frame.copy()
    if "Patient" in export:
        export["Patient"] = export["Patient"].astype(str)
    def spreadsheet_text(value):
        if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "＝", "＋", "－", "＠")):
            return "\t" + value.lstrip("\t")
        return value
    export = export.map(spreadsheet_text)
    return export.rename(columns=column_labels(language)).to_csv(index=False, quoting=csv.QUOTE_NONNUMERIC).encode("utf-8-sig")
