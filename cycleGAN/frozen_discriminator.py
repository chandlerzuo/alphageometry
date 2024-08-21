import torch

class PerplexityCalculator(torch.nn.Module):
    def __init__(self, perplex_model):
        super().__init__()
        self.model = perplex_model
        self.perplexity_criterion = torch.nn.CrossEntropyLoss()

    def resize_token_embeddings(self, logit_len):
        self.model.resize_token_embeddings(logit_len)

    def forward(self, input_logits):
        batch, time_steps, vocab_size = input_logits.size()
        with torch.no_grad():
            # Tokenize the input text and prepare input tensors
            input_ids = torch.argmax(input_logits, dim=-1)

            # Calculate loss without gradient updates
            outputs = self.model(input_ids)

        log_perplexity = self.perplexity_criterion(
            input_logits.view(batch * time_steps, vocab_size),
            torch.argmax(outputs.logits.detach(), dim=-1).view(batch * time_steps,))

        return log_perplexity


if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
    # Example usage
    perplexity_calculator = PerplexityCalculator()

    # Initialize and load the tokenizer and model
    model_name = 'gpt2'
    config = GPT2Config()
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel(config)
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    with torch.no_grad():
        text1 = "The quick brown fox jumps over the lazy dog."
        tokens_tensor = tokenizer.encode(text1, return_tensors='pt')
        outputs = model(tokens_tensor)
        print(f"Log perplexity of text1 generated logits: {perplexity_calculator(outputs.logits)}")


