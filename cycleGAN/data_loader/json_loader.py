import json
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class NLFLDatasetFromJSONL(Dataset):
    def __init__(self, jsonl_file, split='train', overfitting=False, test_size=0.1, random_state=42):
        self.overfitting = overfitting

        # Load the dataset from the JSONL file
        data = self.load_jsonl(jsonl_file)

        # Split the data into train and temporary data (test + validation)
        train_data, temp_data = train_test_split(data, test_size=test_size * 2, random_state=random_state)

        # Split the temporary data into test and validation
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)

        # Assign data according to the split parameter
        if split == 'train' or self.overfitting:
            self.data = train_data
        elif split == 'validation':
            self.data = validation_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError("Split must be 'train', 'validation', or 'test'")

    def load_jsonl(self, file_path):
        data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.overfitting:
            idx = idx % 1  # if overfitting return the same samples
        # Extract formal and natural text from the DataFrame
        formal_text = self.data.iloc[idx]['fl_statement']
        natural_text = self.data.iloc[idx]['nl_statement']
        return {'formal': formal_text, 'natural': natural_text}


if __name__ == '__main__':
    from torch.utils.data import  DataLoader
    data_path = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/rephrased-nl_fl_dataset_all.jsonl'
    # Example usage
    train_dataset = NLFLDatasetFromJSONL(data_path, split='train')
    validation_dataset = NLFLDatasetFromJSONL(data_path, split='validation')
    test_dataset = NLFLDatasetFromJSONL(data_path, split='test')

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example of iterating over Train DataLoader
    for batch in train_loader:
        formal_texts = batch['formal']
        natural_texts = batch['natural']
        print(f"Train Batch - Formal: {formal_texts[0]}, Natural: {natural_texts[0]}")
        break
