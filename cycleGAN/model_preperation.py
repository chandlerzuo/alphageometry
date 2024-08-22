import torch
from frozen_discriminator import PerplexityCalculator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoModelForCausalLM


class AutoEncoderLLM(torch.nn.Module):
    def __init__(self, model_name, encoder, decoder, perplexity_calculator, padding_token_id):
        super().__init__()
        self.padding_token_id = padding_token_id
        self.encoder = encoder
        self.decoder = decoder
        self.model_name = model_name
        # prepare perplexity calculator. keep it frozen!
        self.perplexity_calculator = perplexity_calculator
        # Ensure perplexity_calculator remains frozen
        if perplexity_calculator is not None:
            for param in self.perplexity_calculator.parameters():
                param.requires_grad = False

    def _encode(self, **enc_inputs):
        assert self.encoder is not None
        return self.encoder(**enc_inputs)

    def _decode(self, **decoder_inps):
        return self.decoder(**decoder_inps)

    def forward(self, recon_target, encoder_target, decode_natural=None, **enc_inputs):
        """
        enc_input: Serves as the input to the Autoencode. In ususal operation mode i.e., when bot encoder and the
        decoder are present this is a formal text. This goes into encoder that transforms it into natural language and
        then teh decoder turn it into a formal language.
        In case of decoder only operation, the encoder_target is assumed to be the natural language input
        """
        if self.encoder is None:
            encoder_outputs = None
            decoder_outputs = self._decode(**encoder_target, labels=recon_target)
        else:
            encoder_outputs = self._encode(**enc_inputs, labels=encoder_target)
            decoder_outputs = self._decode(inputs_embeds=encoder_outputs.hidden_states[-1], labels=recon_target)
        if self.perplexity_calculator is None:
            log_perplexity_loss = 0
        else:
            # self.perplexity_calculator.eval()  # should be but deepspeed complains!
            # TODO: Also remove the leading <w> tokens? because perplexity on those tokens are weird to compute?
            log_perplexity_loss = self.perplexity_calculator(encoder_outputs.logits)

        if decode_natural is not None:
            decoded_from_natural = self._decode(**decode_natural)
        else:
            decoded_from_natural = None
        return encoder_outputs, decoder_outputs, log_perplexity_loss, decoded_from_natural


def load_model(model_name, wait_token='<w>', use_pretrained=True, use_perplexity_loss=True, decoder_only=False):
    """
    Load a model with the option to initialize with pretrained weights or randomly.

    Args:
    model_name (str): The name of the model to load (e.g., 'gpt2', 'llama-2').
    use_pretrained (bool): Whether to load the model with pretrained weights. If False, initializes with random weights.

    Returns:
    tuple: A tuple containing the tokenizer and the model.
    """
    perplexity_calculator = None
    encoder = None
    if decoder_only:
        assert not use_perplexity_loss, 'No real use case of using perplexity loss when there is no encoder.'
    if "llama" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if use_perplexity_loss:
            perplexity_calculator = PerplexityCalculator(AutoModelForCausalLM.from_pretrained(model_name))
        if use_pretrained:
            if not decoder_only:
                encoder = AutoModelForCausalLM.from_pretrained(model_name)
            decoder = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            # Load model with configuration from a pretrained model but without the pretrained weights
            config = AutoModelForCausalLM.from_pretrained(model_name).config
            if not decoder_only:
                encoder = AutoModelForCausalLM(config)
            decoder = AutoModelForCausalLM(config)
    elif "gpt2" in model_name.lower():  # Default to GPT2
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        if use_perplexity_loss:
            perplexity_calculator = PerplexityCalculator(GPT2LMHeadModel.from_pretrained(model_name))
        if use_pretrained:
            if not decoder_only:
                encoder = GPT2LMHeadModel.from_pretrained(model_name)
            decoder = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            # Initialize GPT2 with random weights using its configuration
            config = GPT2Config()
            if not decoder_only:
                encoder = GPT2LMHeadModel(config)
            decoder = GPT2LMHeadModel(config)
    else:
        raise ValueError("Model name must contain 'llama' or 'gpt2'.")

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([wait_token])  # add a special wait token
    wait_id = tokenizer.convert_tokens_to_ids(wait_token)

    if not decoder_only:
        encoder.resize_token_embeddings(len(tokenizer))
    decoder.resize_token_embeddings(len(tokenizer))

    if perplexity_calculator is not None:
        perplexity_calculator.resize_token_embeddings(len(tokenizer))

    return AutoEncoderLLM(model_name, encoder, decoder, perplexity_calculator, tokenizer.pad_token), tokenizer, wait_id
