import tqdm
import pandas as pd
import os

# Directory containing all the CSV files
directory = '/is/cluster/scratch/pghosh/dataset/alpha_geo/geometry/'

# List to hold all individual DataFrames
dataframes = []

# Iterate over each file in the directory
for filename in tqdm.tqdm(os.listdir(directory)):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)
        dataframes.append(df)

# Concatenate all DataFrames into one
combined_df = pd.concat(dataframes, ignore_index=True)

# Save the combined DataFrame to a new CSV file
destiantion = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv'
combined_df.to_csv(destiantion, index=False)
print(f'File written to {destiantion}')
