import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_path):
    """
    Load the fine-tuned GPT-2 model and tokenizer from the specified path.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return tokenizer, model


def generate_formal_description(tokenizer, model, natural_lang_description, max_length=100):
    """
    Generate formal description from natural language description.

    Args:
    tokenizer (GPT2Tokenizer): The tokenizer for the model.
    model (GPT2LMHeadModel): The fine-tuned GPT-2 model.
    natural_lang_description (str): The natural language description of the geometry problem.
    max_length (int): Maximum length of the output sequence.

    Returns:
    str: The generated formal language description.
    """
    # Encode the input text and add the separator token
    encoded_input = tokenizer.encode(natural_lang_description + ' [SEP]', return_tensors='pt')

    # Generate output using the model
    output_sequences = model.generate(
        input_ids=encoded_input,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    # Decode the output sequence to a string
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    return generated_text


def main():
    # Path where the fine-tuned model and tokenizer are saved
    model_path = './gpt2_finetuned'

    # Load the model and tokenizer
    tokenizer, model = load_model(model_path)

    # Example natural language description
    # natural_lang_description = "the center of the circle that passes through Q, L, & I is F"
    natural_lang_description = "a"

    # Generate the formal description
    formal_description = generate_formal_description(tokenizer, model, natural_lang_description)
    print("Formal Description:", formal_description)


if __name__ == "__main__":
    main()
