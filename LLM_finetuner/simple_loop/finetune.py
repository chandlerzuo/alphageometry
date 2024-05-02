import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


class GPT2FineTuner:
    def __init__(self, model_name, data_file_path, output_dir):
        self.model_name = model_name
        self.data_file_path = data_file_path
        self.output_dir = output_dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        special_tokens_dict = {'sep_token': '[SEP]', 'pad_token': '[PAD]'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def load_dataset(self, block_size=128):
        try:
            dataset = LineByLineTextDataset(
                tokenizer=self.tokenizer,
                file_path=self.data_file_path,
                block_size=block_size
            )
            print(f"Loaded dataset with {len(dataset)} samples.")
            return dataset
        except Exception as e:
            raise Exception(f"Error loading dataset: {e}")

    def setup_trainer(self, train_dataset):
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=300,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=500,
        )
        return Trainer(
            model=self.model,
            args=training_args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
        )

    def train(self):
        train_dataset = self.load_dataset()
        trainer = self.setup_trainer(train_dataset)
        trainer.train()

    def save_model(self):
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)


def main():
    # Constants
    MODEL_NAME = 'gpt2'  # Use GPT-2 model
    DATA_FILE_PATH = 'overfit_data.txt'
    OUTPUT_DIR = f'./{MODEL_NAME}_finetuned'

    # Initialize finetuner
    finetuner = GPT2FineTuner(MODEL_NAME, DATA_FILE_PATH, OUTPUT_DIR)

    # Train model
    finetuner.train()

    # Save the fine-tuned model and tokenizer
    finetuner.save_model()


if __name__ == "__main__":
    main()
