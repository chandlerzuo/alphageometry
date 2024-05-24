import csv
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

    def __len__(self):
        # Count the number of rows in the CSV file
        with open(self.filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            # Subtract 1 to exclude the header row
            return sum(1 for _ in reader) - 1

    def __getitem__(self, idx):
        # Load a specific row from the CSV file
        with open(self.filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Skip the header row
            next(reader)
            for i, row in enumerate(reader):
                if i == idx:
                    # Convert row values to appropriate data types as needed
                    return {
                        'sl_n': int(row['sl_n']),
                        'num_clauses': int(row['num_clauses']),
                        'nl_statement': row['nl_statement'],
                        'fl_statement': row['fl_statement'],
                        'goal_nl': row['goal_nl'],
                        'goal_fl': row['goal_fl'],
                        'gen_seed': int(row['gen_seed'])
                    }
        raise IndexError(f"Index {idx} is out of range.")

# Example usage:
dataset = CustomDataset('data.csv')
print(len(dataset))  # Output: Number of rows in the CSV file
print(dataset[0])    # Output: First row of data
