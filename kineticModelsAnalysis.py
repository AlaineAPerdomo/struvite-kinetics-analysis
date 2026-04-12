import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import re

# =========================================================
# USER SETTINGS
# =========================================================

excel_file = Path("input_datasets") / "AlaineCummulativeValuesStruvite.xlsx"

# Add all sheet names you want to analyze
sheet_names = ["EPSRawData", "SNAHCO3RawData"]

time_col = "Time"

# Replicate columns + analyte-specific correction + fixed normalization
# Mg%  = Mg_avg / 99 * 100
# NH4% = NH4_avg / 73.5 * 100
# PO4 corrected first by * (70.5/0.5), then PO4% = PO4_avg / 465.3 * 100
analytes = {
    "PO4": {
        "rep_cols": ["PO4_1", "PO4_2", "PO4_3"],
        "conversion_factor": 70.5 / 0.5,
        "normalization_value": 465.3
    },
    "Mg": {
        "rep_cols": ["Mg_1", "Mg_2", "Mg_3"],
        "conversion_factor": 1.0,
        "normalization_value": 99
    },
    "NH4": {
        "rep_cols": ["NH4_1", "NH4_2", "NH4_3"],
        "conversion_factor": 1.0,
        "normalization_value": 73.5
    }
}

run_label = "kinetic_model_analysis"
run_timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M")
output_root = Path("output")
existing_versions = []

for path in output_root.glob(f"v*_{run_label}_*"):
    match = re.match(r"v(\d+)_", path.name)
    if match:
        existing_versions.append(int(match.group(1)))

next_version = max(existing_versions, default=0) + 1
run_version = f"v{next_version}"
main_output = output_root / f"{run_version}_{run_label}_{run_timestamp}"
main_output.mkdir(parents=True, exist_ok=True)
summary_output = main_output / "summaries"
summary_output.mkdir(parents=True, exist_ok=True)

# =========================================================
# PLOT STYLE
# =========================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 120
})

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

def linear_regression_transformed(x, y):
    """
    Perform linear regression in transformed space and return:
    slope, intercept, transformed-space R^2, transformed prediction
    """
    m, b = np.polyfit(x, y, 1)
    y_pred = m * x + b

    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

    return m, b, r2, y_pred

def error_metrics_original_space(y_true, y_pred):
    """
    Compute RMSE and RRMSE in ORIGINAL F-space so metrics are
    comparable across all models.
    """
    residuals = y_true - y_pred
    rmse = np.sqrt(np.mean(residuals ** 2))

    mean_y = np.mean(y_true)
    rrmse = (rmse / mean_y) * 100 if mean_y != 0 else np.nan

    return rmse, rrmse

def predict_f_from_model(model_name, df_model, slope, intercept, time_col):
    """
    Convert the fitted linearized model back into predicted F values.
    This lets RMSE / RRMSE be evaluated in the same space for all models.
    """
    t = df_model[time_col].values

    if model_name == "First-Order Model":
        # ln(1 - F) = m t + b  ->  F = 1 - exp(m t + b)
        f_pred = 1 - np.exp(slope * t + intercept)

    elif model_name == "Higuchi Model":
        # F = m sqrt(t) + b
        f_pred = slope * np.sqrt(t) + intercept

    elif model_name == "Elovich Model":
        # F = m ln(t) + b
        f_pred = slope * np.log(t) + intercept

    elif model_name == "Ritger-Peppas Model":
        # log10(F) = m log10(t) + b  ->  F = 10^(m log10(t) + b)
        f_pred = 10 ** (slope * np.log10(t) + intercept)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return f_pred

def process_dataset(excel_file, sheet_name, time_col, analyte_name, analyte_info):
    print(f"\n==============================")
    print(f"PROCESSING SHEET: {sheet_name} | ANALYTE: {analyte_name}")
    print(f"==============================")

    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    rep_cols = analyte_info["rep_cols"]
    conversion_factor = analyte_info["conversion_factor"]
    normalization_value = analyte_info["normalization_value"]

    required_cols = [time_col] + rep_cols
    validate_columns(df, required_cols, sheet_name)

    # Force numeric
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\nColumns found:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df[required_cols].isna().sum())

    # Drop rows missing time
    df = df.dropna(subset=[time_col]).copy()

    # Apply correction replicate-by-replicate
    corrected_cols = []
    for col in rep_cols:
        new_col = f"{col}_corr"
        df[new_col] = pd.to_numeric(df[col], errors="coerce") * conversion_factor
        corrected_cols.append(new_col)

    print(f"\nCorrected replicate values for {analyte_name}:")
    print(df[[time_col] + corrected_cols].to_string(index=False))

    # Average corrected analyte values
    avg_col = f"{analyte_name}_avg"
    df[avg_col] = df[corrected_cols].mean(axis=1, skipna=False).round(8)

    # Percentage column to match Excel logic exactly
    pct_col = f"{analyte_name}_pct"
    df[pct_col] = ((df[avg_col] / normalization_value) * 100).round(8)

    # Fraction released / dissolved for modeling
    df["F"] = (df[pct_col] / 100).round(10)

    print("\nCheck against Excel:")
    print(df[[time_col, avg_col, pct_col, "F"]].to_string(index=False))

    # Keep only rows valid for transforms involving logs
    df_model = df[(df[time_col] > 0) & (df["F"] > 0) & (df["F"] < 1)].copy()

    if df_model.empty:
        raise ValueError(
            f"Sheet '{sheet_name}', analyte '{analyte_name}': no usable rows remain after filtering."
        )

    # =====================================================
    # TRANSFORMED COLUMNS FOR SELECTED MODELS ONLY
    # =====================================================

    df_model["1-F"] = 1 - df_model["F"]

    # First Order
    df_model["ln(1-F)"] = np.log(df_model["1-F"])

    # Higuchi
    df_model["sqrt_t"] = np.sqrt(df_model[time_col])

    # Elovich
    df_model["ln_t"] = np.log(df_model[time_col])

    # Ritger-Peppas
    df_model["log_t"] = np.log10(df_model[time_col])
    df_model["log_F"] = np.log10(df_model["F"])

    check_cols = [
        "ln(1-F)",
        "sqrt_t",
        "ln_t",
        "log_t",
        "log_F"
    ]

    finite_check = np.isfinite(df_model[check_cols]).all().all()
    if not finite_check:
        raise ValueError(
            f"Sheet '{sheet_name}', analyte '{analyte_name}': transformed values contain non-finite numbers."
        )

    print("\nRows used for modeling:")
    print(df_model[[time_col, avg_col, pct_col, "F"]].to_string(index=False))

    return df, df_model, avg_col, pct_col

def fit_models(df_model, time_col):
    models = {
        "First-Order Model": {
            "x": df_model[time_col].values,
            "y": df_model["ln(1-F)"].values,
            "xlabel": "Time",
            "ylabel": "ln(1 − F)"
        },
        "Higuchi Model": {
            "x": df_model["sqrt_t"].values,
            "y": df_model["F"].values,
            "xlabel": "√t",
            "ylabel": "F"
        },
        "Elovich Model": {
            "x": df_model["ln_t"].values,
            "y": df_model["F"].values,
            "xlabel": "ln(t)",
            "ylabel": "F"
        },
        "Ritger-Peppas Model": {
            "x": df_model["log_t"].values,
            "y": df_model["log_F"].values,
            "xlabel": "log(t)",
            "ylabel": "log(F)"
        }
    }

    fit_results = []
    plot_data = {}

    for model_name, info in models.items():
        x = info["x"]
        y = info["y"]

        slope, intercept, r2_transformed, y_pred_transformed = linear_regression_transformed(x, y)

        # Predict back in original F-space
        f_pred = predict_f_from_model(
            model_name=model_name,
            df_model=df_model,
            slope=slope,
            intercept=intercept,
            time_col=time_col
        )

        # Compute fair cross-model error metrics in F-space
        rmse_f, rrmse_f = error_metrics_original_space(
            y_true=df_model["F"].values,
            y_pred=f_pred
        )

        # Report model parameters in physically meaningful terms.
        ritger_peppas_k = np.nan

        if model_name == "First-Order Model":
            param_label = "k (first-order)"
            param_value = -slope
        elif model_name == "Higuchi Model":
            param_label = "k_H (Higuchi constant)"
            param_value = slope
        elif model_name == "Elovich Model":
            param_label = "slope (1/beta)"
            param_value = slope
        elif model_name == "Ritger-Peppas Model":
            param_label = "n (release exponent)"
            param_value = slope
            ritger_peppas_k = 10 ** intercept
        else:
            param_label = "slope"
            param_value = slope

        fit_results.append({
            "Model": model_name,
            "Parameter": param_label,
            "Value": param_value,
            "k (Ritger-Peppas)": ritger_peppas_k,
            "Slope": slope,
            "Intercept": intercept,
            "R^2 (transformed)": r2_transformed,
            "RMSE in F-space": rmse_f,
            "RRMSE in F-space (%)": rrmse_f
        })

        plot_data[model_name] = {
            "x": x,
            "y": y,
            "y_pred": y_pred_transformed,
            "xlabel": info["xlabel"],
            "ylabel": info["ylabel"],
            "slope": slope,
            "intercept": intercept,
            "r2_transformed": r2_transformed,
            "rmse_f": rmse_f,
            "rrmse_f": rrmse_f,
            "f_pred": f_pred
        }

    # Primary ranking: highest transformed R^2, then lowest RMSE in F-space
    summary_df = pd.DataFrame(fit_results).sort_values(
        by=["R^2 (transformed)", "RMSE in F-space"],
        ascending=[False, True]
    )
    return summary_df, plot_data

def make_model_plot(
    x, y, y_pred, xlabel, ylabel, title,
    r2_transformed, rmse_f, rrmse_f, slope, intercept, save_path
):
    order = np.argsort(x)
    x_sorted = x[order]
    y_pred_sorted = y_pred[order]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=60, alpha=0.9, label="Experimental data")
    plt.plot(
        x_sorted,
        y_pred_sorted,
        linewidth=2.2,
        label=f"Best fit: y = {slope:.4f}x + {intercept:.4f}"
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title, weight="bold")
    plt.text(
        0.05, 0.95,
        f"$R^2$ (transformed) = {r2_transformed:.4f}\n"
        f"RMSE (F-space) = {rmse_f:.4f}\n"
        f"RRMSE (F-space) = {rrmse_f:.2f}%",
        transform=plt.gca().transAxes,
        verticalalignment="top"
    )
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def make_dissolution_plot(df, time_col, save_path, dataset_name):
    plt.figure(figsize=(7, 5))
    plt.plot(df[time_col], df["F"], marker="o", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("F")
    plt.title(f"Dissolution Curve — {dataset_name}", weight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def make_release_plot(df, time_col, avg_col, save_path, dataset_name):
    plt.figure(figsize=(7, 5))
    plt.plot(df[time_col], df[avg_col], marker="o", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(avg_col)
    plt.title(f"Release Profile — {dataset_name}", weight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def make_percentage_plot(df, time_col, pct_col, save_path, dataset_name):
    plt.figure(figsize=(7, 5))
    plt.plot(df[time_col], df[pct_col], marker="o", linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(pct_col)
    plt.title(f"Percentage Release — {dataset_name}", weight="bold")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

# =========================================================
# MAIN LOOP
# =========================================================

combined_results = []
best_model_rows = []

for sheet in sheet_names:
    sheet_output = main_output / sheet
    sheet_output.mkdir(exist_ok=True)

    for analyte_name, analyte_info in analytes.items():
        analyte_output = sheet_output / analyte_name
        analyte_output.mkdir(exist_ok=True)

        print(f"\nAnalyzing {analyte_name} in sheet {sheet}")

        # Process analyte
        df_all, df_model, avg_col, pct_col = process_dataset(
            excel_file=excel_file,
            sheet_name=sheet,
            time_col=time_col,
            analyte_name=analyte_name,
            analyte_info=analyte_info
        )

        # Save processed data
        df_all.to_csv(
            analyte_output / f"{sheet}_{analyte_name}_full_processed.csv",
            index=False
        )
        df_model.to_csv(
            analyte_output / f"{sheet}_{analyte_name}_model_data.csv",
            index=False
        )

        # Fit models
        summary_df, plot_data = fit_models(df_model, time_col)

        print(f"\nModel summary for {sheet} | {analyte_name}:")
        print(summary_df.to_string(index=False))

        summary_df.to_csv(
            summary_output / f"{sheet}_{analyte_name}_model_summary.csv",
            index=False
        )

        # Add to combined summary
        temp_summary = summary_df.copy()
        temp_summary.insert(0, "Analyte", analyte_name)
        temp_summary.insert(0, "Dataset", sheet)
        combined_results.append(temp_summary)

        # Save individual model plots
        for model_name, pdata in plot_data.items():
            filename = model_name.lower().replace(" ", "_").replace("-", "_") + ".png"
            save_path = analyte_output / filename

            make_model_plot(
                x=pdata["x"],
                y=pdata["y"],
                y_pred=pdata["y_pred"],
                xlabel=pdata["xlabel"],
                ylabel=pdata["ylabel"],
                title=f"{model_name} — {sheet} — {analyte_name}",
                r2_transformed=pdata["r2_transformed"],
                rmse_f=pdata["rmse_f"],
                rrmse_f=pdata["rrmse_f"],
                slope=pdata["slope"],
                intercept=pdata["intercept"],
                save_path=save_path
            )

        # Save F-based dissolution curve
        make_dissolution_plot(
            df=df_all,
            time_col=time_col,
            save_path=analyte_output / "dissolution_curve.png",
            dataset_name=f"{sheet} — {analyte_name}"
        )

        # Save average release profile
        make_release_plot(
            df=df_all,
            time_col=time_col,
            avg_col=avg_col,
            save_path=analyte_output / "release_profile.png",
            dataset_name=f"{sheet} — {analyte_name}"
        )

        # Save percentage profile
        make_percentage_plot(
            df=df_all,
            time_col=time_col,
            pct_col=pct_col,
            save_path=analyte_output / "percentage_profile.png",
            dataset_name=f"{sheet} — {analyte_name}"
        )

        # Save best model (using current sort order)
        best_row = summary_df.iloc[0].copy()
        best_model_rows.append({
            "Dataset": sheet,
            "Analyte": analyte_name,
            "Best Model": best_row["Model"],
            "Parameter": best_row["Parameter"],
            "Value": best_row["Value"],
            "k (Ritger-Peppas)": best_row["k (Ritger-Peppas)"],
            "Best R^2 (transformed)": best_row["R^2 (transformed)"],
            "RMSE in F-space": best_row["RMSE in F-space"],
            "RRMSE in F-space (%)": best_row["RRMSE in F-space (%)"],
            "Slope": best_row["Slope"],
            "Intercept": best_row["Intercept"]
        })

# =========================================================
# COMBINED OUTPUTS
# =========================================================

if combined_results:
    combined_summary_df = pd.concat(combined_results, ignore_index=True)
    combined_summary_df.to_csv(summary_output / "combined_model_summary.csv", index=False)

    best_models_df = pd.DataFrame(best_model_rows).sort_values(
        by=["Best R^2 (transformed)", "RMSE in F-space"],
        ascending=[False, True]
    )
    best_models_df.to_csv(summary_output / "best_model_by_dataset.csv", index=False)

    print("\n======================================")
    print("COMBINED MODEL SUMMARY")
    print("======================================")
    print(combined_summary_df.to_string(index=False))

    print("\n======================================")
    print("BEST MODEL FOR EACH DATASET / ANALYTE")
    print("======================================")
    print(best_models_df.to_string(index=False))

print(f"\nAll outputs saved in: {main_output.resolve()}")
