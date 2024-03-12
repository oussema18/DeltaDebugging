# Example reference and candidate sentences
from smooth_bleu import bleu
import pandas as pd


def calculate_bleu_for_row(reference_sentence, candidate_sentence):
    # Calculate the smoothed BLEU score
    bleu_score = bleu(
        [reference_sentence], candidate_sentence, smooth=1
    )  # Enable smoothing
    return bleu_score[0] * 100


# Load the Excel file
file_path = (
    "C:/Users/oussama/Desktop/Bachelor_Thesis/Results_1000_functions_PLBART.xlsx"
)
df = pd.read_excel(file_path, sheet_name="Without_Comments", engine="openpyxl")

# Ensure the BLEU scores list is empty
bleu_scores = []

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    reference = row["Comments"]
    candidate = row["Original Prediction"]
    bleu_score = calculate_bleu_for_row(reference, candidate)
    bleu_scores.append(bleu_score)

# Add the BLEU scores as a new column to the DataFrame
df["Smoothed BLEU"] = bleu_scores

# Write the modified DataFrame back to an Excel file
output_file_path = (
    "C:/Users/oussama/Desktop/Bachelor_Thesis/Results_1000_functions_PLBART.xlsx"
)
with pd.ExcelWriter(
    output_file_path, engine="openpyxl", mode="a", if_sheet_exists="new"
) as writer:
    df.to_excel(writer, sheet_name="BLEU Scores", index=False)
