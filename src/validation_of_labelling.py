import sys
from pathlib import Path

# Get the project root relative to the notebook's location
PROJECT_ROOT = Path().resolve().parent  # one level up from the notebook folder

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -------------------------
# Helpers from your project
# -------------------------
from src.paths import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR)


import sys
from pathlib import Path

# Get the project root relative to the notebook's location
PROJECT_ROOT = Path().resolve().parent  # one level up from the notebook folder

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --- Imports ---
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -------------------------
# Helpers from your project
# -------------------------
from src.paths import (
    DATA_DIR, FIGURES_DIR, TABLES_DIR)


def load_model_data():
    """Load core + profile-pic race info; keep females, drop unknown race, flag White."""
   
    df = pd.read_csv(DATA_DIR / "models_measure_with_gender.csv" )
    skincolor_data = pd.read_csv(DATA_DIR / "model_info_from_profilepic.csv")
    df = df.merge(skincolor_data, left_on = "name", right_on="model_name")
    print("Model data with ethnicity data:", df.shape)

    return df

def preprocess_model_data(df):
    df = df.copy()
    df = df.loc[df["face_detected"]]    
    print("Returning df with shape:", df.shape)

    df["is_female_ff"] = (df["consensus_gender"].map({"male": 0, "female": 1}))
    df["is_female_dotcom"] = (df["gender_dotcom"].map({"Male": 0,"Female": 1}))

    df["is_white_ff"] = (df["predicted_race"].map({"White": 1, 
                                                    'Black' : 0 ,
                                                    'East_Asian' : 0, 
                                                    'Indian' : 0, 
                                                    'Middle_Eastern' : 0,
                                                    'Latino_Hispanic' : 0, 
                                                    'Southeast_Asian' : 0}))

    df["is_white_dotcom"] = (df["ethnicity_dotcom"].map({"White": 1, 
                                                        'Black' : 0 ,
                                                        'South Asian' : 0, 
                                                        'Hispanic/Latino' : 0, 
                                                        'Asian' : 0,
                                                        'Middle Eastern' : 0, 
                                                        'Indigenous / Native American' : 0,
                                                        'Pacific Islander' : 0}))

    df = df.dropna(subset=["is_female_ff", "is_female_dotcom",
                            "is_white_ff","is_white_dotcom"])

    print("Validation df with shape:", df.shape)


    return df 

def get_validation_metrics(y_true, y_pred):
    """Return precision, recall, f1 (weighted), accuracy (non-weighted), and N."""
    report = classification_report(y_true, y_pred, output_dict=True)
    metrics = report["weighted avg"]
    N = len(y_true)
    acc = accuracy_score(y_true, y_pred)
    return {
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1": metrics["f1-score"],
        "Accuracy": acc,
        "N": N
    }


def save_validation_table_latex(metrics_gender, metrics_ethnicity):
    """Generate and save LaTeX validation table."""
    latex = r"""
\begin{{table}}[htb]\centering
\begin{{tabular}}{{p{{4cm}}lllll}}
\hline
\textbf{{Validation Task}} & \textbf{{Precision}} & \textbf{{Recall}} & \textbf{{F1 Score}} & \textbf{{Accuracy}} & \textbf{{N}} \\
\hline
Gender (is\_female) & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {} \\
Ethnicity (is\_white) & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {} \\
\hline
\end{{tabular}}
\caption{{\textbf{{Validation results for gender and ethnicity classification.}} The table reports weighted-average Precision, Recall, F1 Score, and non-weighted Accuracy, along with the number of observations (N).}}
\label{{tab:validation_metrics}}
\end{{table}}
""".format(
        metrics_gender["Precision"], metrics_gender["Recall"], metrics_gender["F1"], metrics_gender["Accuracy"], metrics_gender["N"],
        metrics_ethnicity["Precision"], metrics_ethnicity["Recall"], metrics_ethnicity["F1"], metrics_ethnicity["Accuracy"], metrics_ethnicity["N"]
    )

    tex_path = TABLES_DIR / "validation_results.tex"
    tex_path.write_text(latex)
    print(f"LaTeX table saved to {tex_path}")


# === MAIN ===

model_data = load_model_data()
validation_data = preprocess_model_data(model_data)


# ---- Gender validation (is_female) ----
y_true_gender = validation_data["is_female_dotcom"]
y_pred_gender = validation_data["is_female_ff"]
metrics_gender = get_validation_metrics(y_true_gender, y_pred_gender)


print("\nGender validation (is_female):")
print(confusion_matrix(y_true_gender, y_pred_gender))
print(classification_report(y_true_gender, y_pred_gender, digits=3))
print("Accuracy:", accuracy_score(y_true_gender, y_pred_gender))



# ---- Ethnicity validation (is_white) ----
y_true_eth = validation_data["is_white_dotcom"]
y_pred_eth = validation_data["is_white_ff"]
metrics_eth = get_validation_metrics(y_true_eth, y_pred_eth)

print("\nEthnicity validation (is_white):")
print(confusion_matrix(y_true_eth, y_pred_eth))
print(classification_report(y_true_eth, y_pred_eth, digits=3))
print("Accuracy:", accuracy_score(y_true_eth, y_pred_eth))


save_validation_table_latex(metrics_gender, metrics_eth)



