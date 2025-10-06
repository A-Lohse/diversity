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
    DATA_DIR, FIGURES_DIR, TABLES_DIR, ensure_directories_exist
)

from src.analysis import (
    load_core_datasets,
    create_master_dataset,
    prepare_shows_and_measurements_data,
)


def load_model_data():
    """Load core + profile-pic race info; keep females, drop unknown race, flag White."""
    core = load_core_datasets()
    df = create_master_dataset(core)

    skincolor_data = pd.read_csv(DATA_DIR / "model_info_from_profilepic.csv")
    df["model_name"] = df["filename"].apply(lambda x: x.split(".")[0])
    skincolor_data["model_name"] = skincolor_data["image_file"].apply(lambda x: x.split(".")[0])

    df = df.merge(skincolor_data, on="model_name")
    df = df.loc[df["face_detected"]]
    df = df.loc[df["predicted_race"] != "Unknown"]
    

    return df


model_data = load_model_data()
new_model_data = pd.read_csv(DATA_DIR / "models_measure_with_gender.csv")


model_data["is_female"] = (
    model_data["predicted_gender"]
    .map({"Male": 0, "Female": 1})
)


model_data["is_white"] = model_data["predicted_race"].map({"White": 1, 
          'Black' : 0 ,
         'East_Asian' : 0, 
         'Indian' : 0, 
         'Middle_Eastern' : 0,
       'Latino_Hispanic' : 0, 
       'Southeast_Asian' : 0})
    
new_model_data["is_female"] =  new_model_data["gender_dotcom"] .map({"Male": 0,
           "Female": 1,
           'Non-binary': 0,
           "Unknown" : 0})

   
new_model_data["is_white"] =  new_model_data["ethnicity_dotcom"] .map({"White": 1, 
          'Black' : 0 ,
         'South Asian' : 0, 
         'Hispanic/Latino' : 0, 
         'Asian' : 0,
       'Middle Eastern' : 0, 
       'Indigenous / Native American' : 0,
       'Pacific Islander' : 0})


new_model_data = new_model_data.dropna(subset=["is_female", "is_white"])
model_data = model_data.dropna(subset=["is_female", "is_white"])

# Merge on model_id to get overlap
aligned = model_data.merge(
    new_model_data[["model_id", "is_female", "is_white"]],
    on="model_id",
    suffixes=("_pred", "_true")
)


# ---- Gender validation (is_female) ----
y_true = aligned["is_female_true"]
y_pred = aligned["is_female_pred"]
print(f"Overlap size: {len(y_pred)}")


print("\nGender validation (is_female):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
print("Accuracy:", accuracy_score(y_true, y_pred))



# ---- Ethnicity validation (is_white) ----
mask = aligned["is_white_true"].notna() & aligned["is_white_pred"].notna()
y_true = aligned.loc[mask, "is_white_true"]
y_pred = aligned.loc[mask, "is_white_pred"]

print("\nEthnicity validation (is_white):")
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, digits=3))
print("Accuracy:", accuracy_score(y_true, y_pred))
