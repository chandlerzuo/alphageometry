import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class NLFLDatasetFromCSV(Dataset):
    def __init__(self, csv_file, split='train', test_size=0.1, random_state=42):
        # Load the dataset from the CSV file
        data = pd.read_csv(csv_file)

        # Split the data into train and temporary data (test + validation)
        train_data, temp_data = train_test_split(data, test_size=test_size * 2, random_state=random_state)

        # Split the temporary data into test and validation
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)

        # Assign data according to the split parameter
        if split == 'train':
            self.data = train_data
        elif split == 'validation':
            self.data = validation_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError("Split must be 'train', 'validation', or 'test'")

    def __len__(self):
        # Return the total number of samples in the data
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the formal and natural text from the DataFrame
        formal_text = self.data.iloc[idx]['fl_statement']
        natural_text = self.data.iloc[idx]['nl_statement']
        return {'formal': formal_text, 'natural': natural_text}


if __name__ == '__main__':
    data_path = '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv'
    # Example usage
    train_dataset = NLFLDatasetFromCSV(data_path, split='train')
    validation_dataset = NLFLDatasetFromCSV(data_path, split='validation')
    test_dataset = NLFLDatasetFromCSV(data_path, split='test')

    # Create DataLoaders for each split
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Example of iterating over Train DataLoader
    for batch in train_loader:
        formal_texts = batch['formal']
        natural_texts = batch['natural']
        print(f"Train Batch - Formal: {formal_texts}, Natural: {natural_texts}")
