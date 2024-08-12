from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoModelForCausalLM


def load_model(model_name, use_pretrained=True):
    """
    Load a model with the option to initialize with pretrained weights or randomly.

    Args:
    model_name (str): The name of the model to load (e.g., 'gpt2', 'llama-2').
    use_pretrained (bool): Whether to load the model with pretrained weights. If False, initializes with random weights.

    Returns:
    tuple: A tuple containing the tokenizer and the model.
    """
    if "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_pretrained:
            encoder = AutoModelForCausalLM.from_pretrained(model_name)
            decoder = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            # Load model with configuration from a pretrained model but without the pretrained weights
            config = AutoModelForCausalLM.from_pretrained(model_name).config
            encoder = AutoModelForCausalLM(config)
            decoder = AutoModelForCausalLM(config)
    elif "gpt2" in model_name.lower():  # Default to GPT2
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if use_pretrained:
            encoder = GPT2LMHeadModel.from_pretrained(model_name)
            decoder = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            # Initialize GPT2 with random weights using its configuration
            config = GPT2Config()
            encoder = GPT2LMHeadModel(config)
            decoder = GPT2LMHeadModel(config)
    else:
        raise ValueError("Model name must contain 'llama' or 'gpt2'.")

    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, encoder, decoder

