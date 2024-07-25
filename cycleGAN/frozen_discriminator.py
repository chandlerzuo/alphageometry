import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class PerplexityCalculator(torch.nn.Module):
    def __init__(self, model_name='gpt2'):
        super().__init__()
        # Initialize and load the tokenizer and model
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, text):
        # Tokenize the input text and prepare input tensors
        tokens_tensor = self.tokenizer.encode(text, return_tensors='pt')

        # Calculate loss without gradient updates
        # with torch.no_grad():
        outputs = self.model(tokens_tensor, labels=tokens_tensor)
        log_perplexity = outputs[0]

        return log_perplexity


if __name__ == '__main__':
    # Example usage
    perplexity_calculator = PerplexityCalculator()

    text1 = "The quick brown fox jumps over the lazy dog."
    print(f"Perplexity of text1: {perplexity_calculator(text1)}")

    text2 = "Example of a second piece of text to evaluate."
    print(f"Perplexity of text2: {perplexity_calculator(text2)}")
