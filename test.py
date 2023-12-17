import pandas as pd
import ast

# Read the Excel file
data = pd.read_excel("metricsAfterRemovingComments.xlsx")

# Display the contents

input_string = "[(6, 'to'), (7, '_'), (8, 'url'), (10, 'list'), (11, '('), (18, ' '), (19, ' '), (20, ' '), (21, ' '), (22, '\"'), (23, '\"'), (24, '\"'), (26, '-'), (27, '>'), (28, 'list'), (29, '\\n'), (34, 'Convert'), (36, 'XML'), (37, ' '), (40, 'URL'), (41, ' '), (42, 'List'), (43, '.'), (45, ' '), (46, ' '), (47, ' '), (48, ' '), (49, 'From'), (55, ' '), (56, ' '), (100, 'getElementsByTagName'), (101, '('), (107, '\\n'), (108, ' '), (109, ' '), (110, ' '), (111, ' '), (112, ' '), (113, ' '), (114, ' '), (117, ' '), (126, \"'\"), (140, 'rawurl'), (143, '('), (144, 'url')]"
# Removing newline characters for better readability
input_string = input_string.replace("\\n", "")

# Use ast.literal_eval to safely evaluate the string as a Python literal
output_list = ast.literal_eval(input_string)

print(output_list[0])
