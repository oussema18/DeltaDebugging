import pandas as pd
import ast
from BLEU_SCORE import calculate_cosine_similarity
from helper import get_token_deltas
from treeSitter import extract_variables_names, get_function_name
import nltk


def deltas_to_code(d):
    return " ".join([c[1] for c in d])


# Read the Excel file
data = pd.read_excel("C:/Users/oussama/Downloads/metricsAfterRemovingFN(1).xlsx")
new_column_values = [0] * len(data)
for i in range(987, len(data)):
    # Display the contents
    reduced_tokens = ast.literal_eval(data["Reduced Code Tokens"][i])
    summary = data["Comments"][i]
    code = data["Original Code"][i]
    variables_names = get_function_name(code)
    new_column_values[i] = calculate_cosine_similarity(
        deltas_to_code(get_token_deltas(variables_names)), reduced_tokens
    )


df = pd.read_excel("C:/Users/oussama/Downloads/metricsAfterRemovingFN(1).xlsx")

df["Cosine to Function Name"] = new_column_values

# Write the updated DataFrame back to the Excel file
with pd.ExcelWriter(
    "C:/Users/oussama/Downloads/metricsAfterRemovingFN(1).xlsx",
    engine="openpyxl",
    mode="a",
) as writer:  # Use 'openpyxl' engine to append
    df.to_excel(
        writer, index=False, sheet_name="Sheet4"
    )  # Replace 'Sheet1' with your sheet name

print("Column added and values written to Excel file.")
