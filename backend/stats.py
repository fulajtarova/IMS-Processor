"""This module produces statistics and analysis of model performance, including average accuracy and differences from the average configuration."""

import pandas as pd
import os

# classifier_map = {
#     "Logistic Regression": "LR",
#     "Support Vector Classifier": "SVM",
#     "Decision Tree": "DT",
#     "K-Nearest Neighbors": "KNN",
#     "Naive Bayes": "NB",
#     "Random Forest": "RF",
#     "Gradient Boosting": "GB",
#     "XGBoost": "XGB",
#     "Logistic Regression (Eval)": "LR",
#     "Support Vector Classifier (Eval)": "SVM",
#     "Decision Tree (Eval)": "DT",
#     "K-Nearest Neighbors (Eval)": "KNN",
#     "Naive Bayes (Eval)": "NB",
#     "Random Forest (Eval)": "RF",
#     "Gradient Boosting (Eval)": "GB",
#     "XGBoost (Eval)": "XGB",
# }
classifier_map = {
    "Logistic Regression": "LR",
    "Random Forest": "RF",
    "XGBoost": "XGB",
    "Logistic Regression (Eval)": "LR",
    "Random Forest (Eval)": "RF",
    "XGBoost (Eval)": "XGB",
}

dataset_map = {"honey": "H", "vine": "V", "olive_oil": "O"}


def shorten_config(config: str) -> str:
    """
    Shorten the configuration string to a more readable format.
    """
    components = []
    config = config.lower()

    if "rip_remove" in config:
        components.append("rip")
    if "rip_rel" in config:
        components.append("riprel")
    if "cutoff" in config:
        components.append("cut")

    if "smoothing" in config:
        if "uniform_filter" in config:
            components.append("uni-fil")
        elif "gaussian_blurring" in config:
            components.append("gauss-bl")
        elif "savitzky_golay" in config:
            components.append("savgol")

    if "average" in config:
        components.append("avg")

    if "transformation" in config:
        if "zscore" in config:
            components.append("zscore")
        elif "min_max" in config:
            components.append("minmax")
        elif "max_abs" in config:
            components.append("maxabs")
        elif "robust" in config:
            components.append("robust")

    if "feature_selection_filter" in config:
        if "mutual_info" in config:
            components.append("mi-filter")
        elif "variance_threshold" in config:
            components.append("var-thresh")

    if "feature_selection_wrapper" in config:
        if "support vector machine" in config:
            components.append("rfe-svm")
        elif "logistic regression" in config:
            components.append("rfe-lr")
        elif "random forest" in config:
            components.append("rfe-rf")
        elif "gradient boosting" in config:
            components.append("rfe-gb")

    if "dimensionality_reduction" in config:
        if "lda" in config and "pca" in config:
            components.append("pca-lda")
        elif "lda" in config:
            components.append("lda")
        elif "pca" in config:
            components.append("pca")

    return "_".join(components) if components else "base"


def compute_model_avg(accs):
    """
    Compute the average accuracy for a model, excluding any 1.0 values.
    """
    valid_accs = [a for a in accs if a < 1.0]
    return round(sum(valid_accs) / len(valid_accs), 4) if valid_accs else None


def analyze_model_performance(input_csv_path, output_csv_path):
    """
    Analyze model performance from a CSV file and save the results to another CSV file.
    """
    df = pd.read_csv(input_csv_path, skip_blank_lines=False)

    results = []
    current_config = None
    temp_rows = []

    def process_block(config, rows):
        config_df = pd.DataFrame(rows, columns=df.columns)
        model_avgs = {}
        overfit_flags = []

        models_in_data = config_df["Model"].unique()
        eval_models = {
            m: classifier_map[m]
            for m in models_in_data
            if "(Eval)" in m and m in classifier_map
        }

        for full_model, short_model in eval_models.items():
            model_df = config_df[config_df["Model"] == full_model]
            accs = model_df["Accuracy"].astype(float).tolist()
            if len(accs) == 3:
                avg_acc = compute_model_avg(accs)
                model_avgs[short_model] = avg_acc
                # Report only datasets where accuracy == 1.0
                overfit_datasets = [
                    dataset_map.get(
                        model_df.iloc[i]["Dataset"], model_df.iloc[i]["Dataset"]
                    )
                    for i, a in enumerate(accs)
                    if a == 1.0
                ]

                if overfit_datasets:
                    overfit_flags.append(
                        f"{short_model}: {', '.join(overfit_datasets)}"
                    )

        model_values = [model_avgs.get(m) for m in eval_models.values()]
        overall_avg = (
            round(sum(model_values) / len(model_values), 4) if model_values else None
        )
        overfit_summary = " | ".join(overfit_flags) if overfit_flags else "No"

        results.append(
            {
                "Config": shorten_config(config),
                **{
                    short_model: model_avgs.get(short_model, "")
                    for short_model in eval_models.values()
                },
                "Avg": overall_avg,
                "Overfit": overfit_summary,
            }
        )

    for _, row in df.iterrows():
        if pd.isna(row["Dataset"]):
            continue
        if pd.isna(row["Accuracy"]):
            if current_config and temp_rows:
                process_block(current_config, temp_rows)
            current_config = row["Dataset"]
            temp_rows = []
        else:
            temp_rows.append(row)

    if current_config and temp_rows:
        process_block(current_config, temp_rows)

    output_df = pd.DataFrame(results)
    output_df = output_df.sort_values(by="Avg", ascending=False).reset_index(drop=True)
    output_df.insert(0, "#", output_df.index + 1)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Analysis saved to: {output_csv_path}")


def calculate_differences_from_avg(input_csv_path, output_csv_path):
    """
    Calculate differences from the average configuration and save to a new CSV file.
    """
    df = pd.read_csv(input_csv_path)

    avg_row = df[df["Config"] == "avg"].iloc[0]
    avg_percentages = avg_row.copy()
    model_columns = [col for col in df.columns if col not in ["#", "Config", "Overfit"]]
    for model in model_columns:
        avg_percentages[model] = (avg_row[model] * 100).round(2)

    differences = df.copy()
    for model in model_columns:
        differences[model] = ((df[model] - avg_row[model]) * 100).round(2)

    differences = differences[differences["Config"] != "avg"]
    differences = differences.sort_values(by="Avg", ascending=False)

    avg_row_with_percentages = avg_percentages.to_frame().T
    avg_row_with_percentages["Config"] = "avg (baseline)"

    if "Overfit" in differences.columns:
        overfit_column = differences.pop("Overfit")
        differences["Overfit"] = overfit_column

    result = pd.concat([avg_row_with_percentages, differences], ignore_index=True)

    if "#" in result.columns:
        result = result.drop(columns=["#"])

    result.insert(0, "#", [0] + list(range(1, len(result))))

    result.to_csv(output_csv_path, index=False, quoting=1, quotechar='"')
    print(f"Differences from 'avg' configuration saved to: {output_csv_path}")


def generate_latex_from_csv(
    csv_path,
    caption="Model Accuracy Changes Compared to Baseline (Percentage Accuracy)",
    label="tab:model_accuracy_changes",
):
    """
    Generate LaTeX table from CSV file.
    """

    df = pd.read_csv(csv_path)

    df["Config"] = df["Config"].str.replace("_", r"\_", regex=False)

    non_model_cols = ["#", "Config", "Overfit"]
    model_cols = [col for col in df.columns if col not in non_model_cols]
    df[model_cols] = df[model_cols].astype(object)

    def format_value(x, is_baseline=False):
        if pd.isna(x):
            return ""
        if abs(float(x)) < 1e-6:
            return "–"
        val = int(round(float(x)))
        return f"{val}" if is_baseline else f"{'+' if val > 0 else '–'}{abs(val)}"

    for col in model_cols:
        df.loc[1:, col] = df.loc[1:, col].apply(lambda x: format_value(x, False))
        df.loc[0, col] = format_value(df.loc[0, col], True)

    df.columns = [f"{col}" if col in model_cols else col for col in df.columns]

    latex_cols = ["#", "Config"] + [f"{col}" for col in model_cols]
    df = df[latex_cols]

    body = df.to_latex(index=False, escape=False, header=True)

    lines = body.splitlines()
    header = "\n".join(lines[2:5])
    data_rows = "\n".join(lines[5:-2])
    bottomrule = lines[-2]

    col_format = "|r|l|" + "|c" * len(model_cols) + "|"
    full_latex = f"""
                \\begin{{tabular}}{{{col_format}}}
                \\hline
                {header}
                \\hline
                {data_rows}
                \\hline
                {bottomrule}
                \\end{{tabular}}
                """.strip()
    return full_latex


if __name__ == "__main__":

    output_dir = r"C:\Users\Laura\Documents\laura\school\BP\vysledky"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Analyze model performance
    analyze_model_performance(
        os.path.join(
            output_dir,
            "kombinacie.csv",
        ),
        os.path.join(
            output_dir,
            "kombinacie_simple.csv",
        ),
    )
    # 2. Calculate differences from avg
    calculate_differences_from_avg(
        os.path.join(
            output_dir,
            "kombinacie_simple.csv",
        ),
        os.path.join(
            output_dir,
            "kombinacie_diff.csv",
        ),
    )
    # 3. Generate LaTeX from CSV
    latex_output = generate_latex_from_csv(
        os.path.join(
            output_dir,
            "kombinacie_diff.csv",
        ),
        caption="Model Accuracy Changes Compared to Baseline (Percentage Accuracy)",
        label="tab:model_accuracy_changes",
    )
    print(latex_output)
