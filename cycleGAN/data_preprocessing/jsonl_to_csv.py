import json
import os
import pandas as pd
import tqdm
import csv  # Correct module for quoting

# Path to the folder with JSONL files
folder_path = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/rephrases-10k'

# Output CSV file
output_csv = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/rephrases-10k.csv'

# List to store data for all rows
data_rows = []


# Function to calculate the number of tokens
def count_words(*statements):
    return sum(len(statement.split()) for statement in statements)


# Loop through each file in the folder
for filename in tqdm.tqdm(os.listdir(folder_path)):
    if filename.endswith('.jsonl'):
        file_path = os.path.join(folder_path, filename)

        # Open and read the JSONL file
        with open(file_path, mode='r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                line_frm_file_i = json.loads(line)

                # Create the new fields
                nl_statement = f"{line_frm_file_i['nl_statement']} Then prove that {line_frm_file_i['goal_nl']}"
                fl_statement = f"{line_frm_file_i['fl_statement']} ? {line_frm_file_i['goal_fl']}"
                rephrase = f"{line_frm_file_i['rephrase']} Prove that {line_frm_file_i['goal_nl']}"

                # Calculate the total tokens
                total_word_count = count_words(nl_statement, rephrase, fl_statement,
                                               line_frm_file_i['goal_nl'])

                # Append data as a row
                data_rows.append({
                    'nl_statement': nl_statement,
                    'rephrase': rephrase,
                    'fl_statement': fl_statement,
                    'total_token_lens': total_word_count
                })

# Create a DataFrame from the collected data
df = pd.DataFrame(data_rows)

# Write the DataFrame to CSV with quotes around all text fields
# df.to_csv(output_csv, index=False, quoting=pd.io.common.csv.QUOTE_ALL)
df.to_csv(output_csv, index=False, quoting=csv.QUOTE_ALL)

print(f"Data merged and written to {output_csv} successfully!")
