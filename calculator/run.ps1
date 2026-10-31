param([string]$Python = "python", [ValidateSet("en", "zh")][string]$Language = "en")
Set-Location -LiteralPath $PSScriptRoot
$entry = if ($Language -eq "zh") { "app_zh.py" } else { "app.py" }
& $Python -m streamlit run $entry
