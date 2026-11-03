"""Input units and checks shared by the two interfaces."""
from math import isfinite
from numbers import Real

FIELDS = {
    "age": ("Age", "年龄", "years", "岁", 18, 120, 70),
    "sex": ("Sex", "性别", "", "", 0, 1, 1),
    "scr": ("Serum creatinine", "血清肌酐", "μmol/L", "μmol/L", 10, 3000, 100),
    "hba1c": ("HbA1c", "糖化血红蛋白", "%", "%", 2, 25, 8),
    "hemoglobin": ("Hemoglobin", "血红蛋白", "g/L", "g/L", 20, 250, 120),
    "uric_acid": ("Serum uric acid", "血清尿酸", "μmol/L", "μmol/L", 1, 2500, 350),
    "albumin": ("Serum albumin", "血清白蛋白", "g/L", "g/L", 1, 70, 35),
    "alp": ("Alkaline phosphatase", "碱性磷酸酶", "U/L", "U/L", 1, 3000, 90),
    "ada": ("Adenosine deaminase", "腺苷脱氨酶", "U/L", "U/L", 0, 300, 15),
}


def normalize(values, scr_unit="umol", hb_unit="gL"):
    """Convert creatinine and hemoglobin to the model's units; preserve missing values."""
    result = {key: values.get(key) for key in FIELDS}
    if result["scr"] is not None and scr_unit == "mg":
        result["scr"] *= 88.4
    if result["hemoglobin"] is not None and hb_unit == "gdL":
        result["hemoglobin"] *= 10
    return result


def check_inputs(values, language="en"):
    """Return blocking errors and confirmable range warnings in normalized units."""
    en = language == "en"
    bad, soft = {}, {}
    for key, field in FIELDS.items():
        value = values[key]
        if not isinstance(value, Real) or isinstance(value, bool) or not isfinite(value):
            bad[key] = "Required" if en else "请填写或选择此项"
        elif key == "sex":
            if value not in (0, 1):
                bad[key] = "Select sex" if en else "请选择性别"
        elif value < 0 or (key != "ada" and value == 0):
            bad[key] = "Check the value and unit" if en else "请核对数值及单位"
        elif key == "age" and not 18 <= value <= 120:
            bad[key] = "Enter an age of 18–120 years" if en else "请输入18–120岁的年龄"
        elif key == "hba1c" and value > 100:
            bad[key] = "HbA1c cannot exceed 100%" if en else "糖化血红蛋白不能超过100%"
        elif not field[4] <= value <= field[5]:
            label, unit = field[0 if en else 1], field[2 if en else 3]
            soft[key] = f"{label}: {'check range' if en else '请核对范围'} {field[4]}–{field[5]} {unit}"
    return bad, soft
