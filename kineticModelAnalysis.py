import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# USER SETTINGS
# =========================================================

excel_file = Path("input_datasets") / "AlaineCummulativeValuesStruvite.xlsx"
sheet_names = ["EPSRawData", "SNAHCO3RawData"]
time_col = "Time"

# Mg%  = Mg_avg / 99 * 100
# NH4% = NH4_avg / 73.5 * 100
# PO4 corrected first by * (70.5/0.5), then PO4% = PO4_avg / 465.3 * 100
analytes = {
    "PO4": {
        "rep_cols": ["PO4_1", "PO4_2", "PO4_3"],
        "conversion_factor": 70.5 / 0.5,
        "normalization_value": 465.3,
    },
    "Mg": {
        "rep_cols": ["Mg_1", "Mg_2", "Mg_3"],
        "conversion_factor": 1.0,
        "normalization_value": 99,
    },
    "NH4": {
        "rep_cols": ["NH4_1", "NH4_2", "NH4_3"],
        "conversion_factor": 1.0,
        "normalization_value": 73.5,
    },
}

run_label = "kinetic_model_analysis"
run_timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M")
output_root = Path("output")


# =========================================================
# OUTPUT SETUP
# =========================================================

def next_run_directory(root: Path, label: str, timestamp: str) -> Path:
    existing_versions = []

    for path in root.glob(f"v*_{label}_*"):
        match = re.match(r"v(\d+)_", path.name)
        if match:
            existing_versions.append(int(match.group(1)))

    next_version = max(existing_versions, default=0) + 1
    run_version = f"v{next_version}"
    return root / f"{run_version}_{label}_{timestamp}"


main_output = next_run_directory(output_root, run_label, run_timestamp)
origin_ready_output = main_output / "origin_ready"
summary_output = main_output / "summary"

origin_ready_output.mkdir(parents=True, exist_ok=True)
summary_output.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def validate_columns(df, required_cols, sheet_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(
            f"Sheet '{sheet_name}' is missing required columns: {missing}\n"
            f"Expected columns: {required_cols}"
        )


def dataset_label(sheet_name):
    return sheet_name.replace("RawData", "").rstrip("_")


def linear_regression_transformed(x, y):
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b

    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return m, b, r2, y_pred


def error_metrics_original_space(y_true, y_pred):
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals**2))

    mean_y = np.mean(y_true)
    rrmse = (rmse / mean_y) * 100 if mean_y != 0 else np.nan

    return rmse, rrmse


def predict_f_from_model(model_name, df_model, slope, intercept):
    t = df_model[time_col].values

    if model_name == "First-Order":
        f_pred = 1 - np.exp(slope * t + intercept)
    elif model_name == "Higuchi":
        f_pred = slope * np.sqrt(t) + intercept
    elif model_name == "Elovich":
        f_pred = slope * np.log(t) + intercept
    elif model_name == "RitgerPeppas":
        f_pred = 10 ** (slope * np.log10(t) + intercept)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return f_pred


def process_dataset(excel_path, sheet_name, analyte_name, analyte_info):
    print(f"\n==============================")
    print(f"PROCESSING SHEET: {sheet_name} | ANALYTE: {analyte_name}")
    print(f"==============================")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    rep_cols = analyte_info["rep_cols"]
    conversion_factor = analyte_info["conversion_factor"]
    normalization_value = analyte_info["normalization_value"]

    required_cols = [time_col] + rep_cols
    validate_columns(df, required_cols, sheet_name)

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[time_col]).copy()
    df = df.sort_values(time_col).reset_index(drop=True)

    corrected_cols = []
    for col in rep_cols:
        corrected_col = f"{col}_corr"
        df[corrected_col] = df[col] * conversion_factor
        corrected_cols.append(corrected_col)

    avg_col = f"{analyte_name}_avg"
    pct_col = f"{analyte_name}_pct"

    df[avg_col] = df[corrected_cols].mean(axis=1, skipna=False).round(8)
    df[pct_col] = ((df[avg_col] / normalization_value) * 100).round(8)
    df["F"] = (df[pct_col] / 100).round(10)

    df_positive_time = df[df[time_col] > 0].copy()
    df_positive_time["ln_t"] = np.log(df_positive_time[time_col])
    df_positive_time["sqrt_t"] = np.sqrt(df_positive_time[time_col])

    df_model = df[(df[time_col] > 0) & (df["F"] > 0) & (df["F"] < 1)].copy()
    if df_model.empty:
        raise ValueError(
            f"Sheet '{sheet_name}', analyte '{analyte_name}': no usable rows remain after filtering."
        )

    df_model["1-F"] = 1 - df_model["F"]
    df_model["ln(1-F)"] = np.log(df_model["1-F"])
    df_model["sqrt_t"] = np.sqrt(df_model[time_col])
    df_model["ln_t"] = np.log(df_model[time_col])
    df_model["log_t"] = np.log10(df_model[time_col])
    df_model["log_F"] = np.log10(df_model["F"])

    check_cols = ["ln(1-F)", "sqrt_t", "ln_t", "log_t", "log_F"]
    finite_check = np.isfinite(df_model[check_cols]).all().all()
    if not finite_check:
        raise ValueError(
            f"Sheet '{sheet_name}', analyte '{analyte_name}': transformed values contain non-finite numbers."
        )

    return {
        "all_data": df,
        "positive_time_data": df_positive_time,
        "model_data": df_model,
        "avg_col": avg_col,
        "pct_col": pct_col,
    }


def fit_models(df_model):
    models = {
        "First-Order": {
            "x": df_model[time_col].values,
            "y": df_model["ln(1-F)"].values,
            "x_label": "Time",
            "y_label": "ln(1-F)",
        },
        "Higuchi": {
            "x": df_model["sqrt_t"].values,
            "y": df_model["F"].values,
            "x_label": "sqrt(t)",
            "y_label": "F",
        },
        "Elovich": {
            "x": df_model["ln_t"].values,
            "y": df_model["F"].values,
            "x_label": "ln(t)",
            "y_label": "F",
        },
        "RitgerPeppas": {
            "x": df_model["log_t"].values,
            "y": df_model["log_F"].values,
            "x_label": "log(t)",
            "y_label": "log(F)",
        },
    }

    fit_results = []

    for model_name, info in models.items():
        slope, intercept, r2_transformed, y_pred_transformed = linear_regression_transformed(
            info["x"],
            info["y"],
        )

        f_pred = predict_f_from_model(
            model_name=model_name,
            df_model=df_model,
            slope=slope,
            intercept=intercept,
        )
        rmse_f, rrmse_f = error_metrics_original_space(df_model["F"].values, f_pred)

        reported_parameter = np.nan
        ritger_peppas_k = np.nan

        if model_name == "First-Order":
            parameter_name = "k"
            reported_parameter = -slope
        elif model_name == "Higuchi":
            parameter_name = "k_H"
            reported_parameter = slope
        elif model_name == "Elovich":
            parameter_name = "slope (1/beta)"
            reported_parameter = slope
        else:
            parameter_name = "n"
            reported_parameter = slope
            ritger_peppas_k = 10 ** intercept

        fit_results.append(
            {
                "Model": model_name,
                "Parameter": parameter_name,
                "Value": reported_parameter,
                "k (RitgerPeppas)": ritger_peppas_k,
                "Slope": slope,
                "Intercept": intercept,
                "R^2 (transformed)": r2_transformed,
                "RMSE in F-space": rmse_f,
                "RRMSE in F-space (%)": rrmse_f,
                "Linearized X": info["x_label"],
                "Linearized Y": info["y_label"],
                "Points Used": len(df_model),
                "Linearized_Y_Predicted": [y_pred_transformed],
                "F_Predicted": [f_pred],
            }
        )

    summary_df = pd.DataFrame(fit_results).sort_values(
        by=["R^2 (transformed)", "RMSE in F-space"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return summary_df


def model_export_frames(processed_by_analyte):
    analyte_order = [name for name in analytes.keys() if name in processed_by_analyte]

    elovich_time = pd.DataFrame(
        {
            time_col: sorted(
                {
                    t
                    for analyte_name in analyte_order
                    for t in processed_by_analyte[analyte_name]["positive_time_data"][time_col].tolist()
                }
            )
        }
    )
    higuchi_time = pd.DataFrame(
        {
            time_col: sorted(
                {
                    t
                    for analyte_name in analyte_order
                    for t in processed_by_analyte[analyte_name]["all_data"][time_col].tolist()
                    if pd.notna(t)
                }
            )
        }
    )
    first_order_time = pd.DataFrame(
        {
            time_col: sorted(
                {
                    t
                    for analyte_name in analyte_order
                    for t in processed_by_analyte[analyte_name]["model_data"][time_col].tolist()
                }
            )
        }
    )
    ritger_time = first_order_time.copy()

    elovich_df = elovich_time.copy()
    elovich_df["ln(t)"] = np.log(elovich_df[time_col])

    higuchi_df = higuchi_time.copy()
    higuchi_df["sqrt(t)"] = np.sqrt(higuchi_df[time_col])

    first_order_df = first_order_time.copy()
    ritger_df = ritger_time.copy()
    ritger_df["log(t)"] = np.log10(ritger_df[time_col])

    for analyte_name in analyte_order:
        avg_col = processed_by_analyte[analyte_name]["avg_col"]

        elovich_source = processed_by_analyte[analyte_name]["positive_time_data"][
            [time_col, avg_col]
        ].rename(columns={avg_col: f"Qt_{analyte_name}"})
        elovich_df = elovich_df.merge(elovich_source, on=time_col, how="left")

        higuchi_source = processed_by_analyte[analyte_name]["all_data"][
            [time_col, avg_col]
        ].rename(columns={avg_col: f"Qt_{analyte_name}"})
        higuchi_df = higuchi_df.merge(higuchi_source, on=time_col, how="left")

        first_order_source = processed_by_analyte[analyte_name]["model_data"][
            [time_col, "ln(1-F)"]
        ].rename(columns={"ln(1-F)": f"ln(1-F)_{analyte_name}"})
        first_order_df = first_order_df.merge(first_order_source, on=time_col, how="left")

        ritger_source = processed_by_analyte[analyte_name]["model_data"][
            [time_col, "log_F"]
        ].rename(columns={"log_F": f"log(F)_{analyte_name}"})
        ritger_df = ritger_df.merge(ritger_source, on=time_col, how="left")

    first_order_columns = [time_col] + [f"ln(1-F)_{name}" for name in analyte_order]
    ritger_columns = [time_col, "log(t)"] + [f"log(F)_{name}" for name in analyte_order]
    elovich_columns = [time_col, "ln(t)"] + [f"Qt_{name}" for name in analyte_order]
    higuchi_columns = [time_col, "sqrt(t)"] + [f"Qt_{name}" for name in analyte_order]

    return {
        "Elovich": elovich_df[elovich_columns],
        "Higuchi": higuchi_df[higuchi_columns],
        "FirstOrder": first_order_df[first_order_columns],
        "RitgerPeppas": ritger_df[ritger_columns],
    }


def compact_fit_summary(dataset_name, fit_summary_by_analyte):
    summary_frames = []
    best_rows = []

    for analyte_name, fit_df in fit_summary_by_analyte.items():
        analyte_summary = fit_df.drop(columns=["Linearized_Y_Predicted", "F_Predicted"]).copy()
        analyte_summary.insert(0, "Analyte", analyte_name)
        analyte_summary.insert(0, "Dataset", dataset_name)
        analyte_summary["Best Model Rank"] = range(1, len(analyte_summary) + 1)
        summary_frames.append(analyte_summary)
        best_rows.append(analyte_summary.iloc[[0]].copy())

    combined = pd.concat(summary_frames, ignore_index=True)
    best = pd.concat(best_rows, ignore_index=True)
    return combined, best


def write_run_readme(output_path: Path, dataset_files, summary_files):
    lines = [
        "Kinetic model analysis output",
        "",
        "This run is organized to reduce clutter.",
        "",
        "origin_ready/",
        "  One workbook per dataset, with one sheet per Origin-ready model export:",
        "  Elovich, Higuchi, FirstOrder, RitgerPeppas",
        "",
        "summary/",
        "  Compact fit summaries only. No per-analyte CSV/image clutter.",
        "",
        "Dataset workbooks:",
    ]

    for path in dataset_files:
        lines.append(f"  - {path.relative_to(output_path)}")

    lines.append("")
    lines.append("Summary files:")
    for path in summary_files:
        lines.append(f"  - {path.relative_to(output_path)}")

    output_path.joinpath("README.txt").write_text("\n".join(lines), encoding="utf-8")


# =========================================================
# MAIN LOOP
# =========================================================

dataset_workbooks = []
summary_files = []
combined_summary_parts = []
best_model_parts = []

for sheet in sheet_names:
    dataset_name = dataset_label(sheet)
    processed_by_analyte = {}
    fit_summary_by_analyte = {}

    for analyte_name, analyte_info in analytes.items():
        processed = process_dataset(
            excel_path=excel_file,
            sheet_name=sheet,
            analyte_name=analyte_name,
            analyte_info=analyte_info,
        )
        fit_summary = fit_models(processed["model_data"])

        processed_by_analyte[analyte_name] = processed
        fit_summary_by_analyte[analyte_name] = fit_summary

        print(f"\nModel summary for {sheet} | {analyte_name}:")
        print(
            fit_summary.drop(columns=["Linearized_Y_Predicted", "F_Predicted"]).to_string(index=False)
        )

    model_frames = model_export_frames(processed_by_analyte)
    dataset_summary_df, dataset_best_df = compact_fit_summary(dataset_name, fit_summary_by_analyte)

    combined_summary_parts.append(dataset_summary_df)
    best_model_parts.append(dataset_best_df)

    dataset_workbook_path = origin_ready_output / f"{dataset_name}_Kinetics.xlsx"
    with pd.ExcelWriter(dataset_workbook_path, engine="openpyxl") as writer:
        for sheet_title, export_df in model_frames.items():
            export_df.to_excel(writer, sheet_name=sheet_title, index=False)

    dataset_summary_path = summary_output / f"{dataset_name}_FitSummary.xlsx"
    with pd.ExcelWriter(dataset_summary_path, engine="openpyxl") as writer:
        dataset_summary_df.to_excel(writer, sheet_name="ModelFits", index=False)
        dataset_best_df.to_excel(writer, sheet_name="BestModels", index=False)

    dataset_workbooks.append(dataset_workbook_path)
    summary_files.append(dataset_summary_path)

if combined_summary_parts:
    combined_summary_df = pd.concat(combined_summary_parts, ignore_index=True)
    best_models_df = pd.concat(best_model_parts, ignore_index=True).sort_values(
        by=["R^2 (transformed)", "RMSE in F-space"],
        ascending=[False, True],
    ).reset_index(drop=True)

    combined_summary_path = summary_output / "Combined_FitSummary.xlsx"
    with pd.ExcelWriter(combined_summary_path, engine="openpyxl") as writer:
        combined_summary_df.to_excel(writer, sheet_name="AllModelFits", index=False)
        best_models_df.to_excel(writer, sheet_name="BestModels", index=False)

    combined_csv_path = summary_output / "best_model_by_dataset.csv"
    best_models_df.to_csv(combined_csv_path, index=False)

    summary_files.extend([combined_summary_path, combined_csv_path])

    print("\n======================================")
    print("COMBINED MODEL SUMMARY")
    print("======================================")
    print(combined_summary_df.to_string(index=False))

    print("\n======================================")
    print("BEST MODEL FOR EACH DATASET / ANALYTE")
    print("======================================")
    print(best_models_df.to_string(index=False))

write_run_readme(main_output, dataset_workbooks, summary_files)

print(f"\nAll outputs saved in: {main_output.resolve()}")
