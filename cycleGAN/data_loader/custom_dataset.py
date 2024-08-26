#%%
import itertools
import json
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# # dataset that combines
# class MergeDataset(Dataset):

class CustomDataset(Dataset):
    def __init__(self, filename, split='train', overfitting=False, test_size=0.1, random_state=42, nrows=None):
        self.overfitting = overfitting

        data = self.load_file(filename, nrows=nrows)
        
        # Split the data into train and temporary data (test + validation)
        train_data, temp_data = train_test_split(data, test_size=test_size * 2, random_state=random_state)

        # Split the temporary data into test and validation
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=random_state)

        # Assign data according to the split parameter
        if split == 'train' or self.overfitting:  # If overfitting, use the train data for all splits
            self.data = train_data
        elif split == 'validation':
            self.data = validation_data
        elif split == 'test':
            self.data = test_data
        else:
            raise ValueError("Split must be 'train', 'validation', or 'test'")

    @staticmethod
    def load(filename, *args, **kwargs):
        DatasetClass = NLFLDatasetFromCSV if str(filename).endswith('.csv') else NLFLDatasetFromJSONL
        return DatasetClass(filename, *args, **kwargs)
    
    def load_file(self, filename, nrows=None):
        raise NotImplementedError
    
    def __len__(self):
        # Return the total number of samples in the data
        return len(self.data)

    def __getitem__(self, idx):
        # Fetch the formal and natural text from the DataFrame
        if self.overfitting:
            idx = 0 # always return the same batch
        # print(f"Rank {os.environ.get('LOCAL_RANK', '0')}, Index: {idx}")
        formal_text = self.data.iloc[idx]['fl_statement']
        natural_text = self.data.iloc[idx]['nl_statement']
        return {'formal': formal_text, 'natural': natural_text}

def load_jsonl(file_path, nrows=None):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in itertools.islice(file, nrows):
            data.append(json.loads(line))
    return pd.DataFrame(data)
class NLFLDatasetFromCSV(CustomDataset):
    def load_file(self, filename, nrows=None):
        return pd.read_csv(filename, nrows=nrows)
class NLFLDatasetFromJSONL(CustomDataset):
    def load_file(self, filename, nrows=None):
        return load_jsonl(filename, nrows=nrows)
    
#%%
if __name__ == '__main__':
    for data_path in [
        '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv', 
        '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/rephrased-nl_fl_dataset_all.jsonl'
    ]:
    
        # Example usage
        train_dataset = CustomDataset.load(data_path, split='train')
        validation_dataset = CustomDataset.load(data_path, split='validation')
        test_dataset = CustomDataset.load(data_path, split='test')

        # Create DataLoaders for each split
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Example of iterating over Train DataLoader
        for batch in train_loader:
            formal_texts = batch['formal']
            natural_texts = batch['natural']
            print(f"Train Batch - Formal: {formal_texts}, Natural: {natural_texts[0]}")
            break
# %%
