#%%
import functools
import os
from pathlib import Path
import tempfile
import torch
from utils import create_dir, is_frozen, load_pretrained_config_from_scratch, save_model, save_tokenizer
from frozen_discriminator import PerplexityCalculator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoModelForCausalLM, AutoModel
import textwrap

class AutoEncoderLLM(torch.nn.Module):
    """
    Encoder-decoder terminology is reversed from usual one!!!
    Decoder: natural to formal
    Encoder: formal to natural
    """
    def __init__(self, model_name, encoder, decoder, perplexity_calculator, padding_token_id):
        super().__init__()
        self.model_name = model_name
        self.encoder = encoder
        self.decoder = decoder
        self.perplexity_calculator = perplexity_calculator
        self.padding_token_id = padding_token_id
        
        if self.perplexity_calculator is not None:
            assert is_frozen(self.perplexity_calculator.model)
        
    def _encode(self, **enc_inputs):
        assert self.encoder is not None
        return self.encoder(**enc_inputs)
    
    def same_archs(self, other: 'AutoEncoderLLM'):
        """check for equal classes for encoder, decoder, perplexity_calculator"""
        return self.model_name == other.model_name and self.encoder.__class__ == other.encoder.__class__ and self.decoder.__class__ == other.decoder.__class__ and self.perplexity_calculator.__class__ == other.perplexity_calculator.__class__ and self.padding_token_id == other.padding_token_id
    
    def short_info(self):
        # encoder class name, decoder class name, perplexity_calculator class name, padding_token_id
        return textwrap.dedent(f"""\
            Encoder: {self.encoder.__class__.__name__}
            Decoder: {self.decoder.__class__.__name__}
            PerplexityCalculator: {self.perplexity_calculator.__class__.__name__}
            PaddingTokenId: {self.padding_token_id}""")

    def _decode(self, **decoder_inps):
        """run the decoder"""
        assert self.decoder is not None
        return self.decoder(**decoder_inps)

    def forward(self, recon_target, encoder_target, decode_natural=None, **enc_inputs):
        """
        enc_input: Serves as the input to the Autoencode. In ususal operation mode i.e., when bot encoder and the
        decoder are present this is a formal text. This goes into encoder that transforms it into natural language and
        then teh decoder turn it into a formal language.
        In case of decoder only operation, the encoder_target is assumed to be the natural language input
        """
        # import ipdb; ipdb.set_trace()
        if self.encoder is None:  # decoder only model
            encoder_outputs = None
            decoder_outputs = self._decode(**encoder_target, labels=recon_target)
        elif self.decoder is None:  # encoder only model
            decoder_outputs = None
            encoder_outputs = self._encode(**enc_inputs, labels=encoder_target)
        else:  # autoencoder model
            encoder_outputs = self._encode(**enc_inputs, labels=encoder_target)
            decoder_outputs = self._decode(inputs_embeds=encoder_outputs.hidden_states[-1], labels=recon_target)

        if self.perplexity_calculator is None:
            log_perplexity_loss = 0
        else:
            # self.perplexity_calculator.eval()  # should be but deepspeed complains!
            # TODO: Also remove the leading <w> tokens? because perplexity on those tokens are weird to compute?
            log_perplexity_loss = self.perplexity_calculator(encoder_outputs.logits)

        if decode_natural is not None and self.decoder is not None:
            decoded_from_natural = self._decode(**decode_natural)
        else:
            decoded_from_natural = None
        return encoder_outputs, decoder_outputs, log_perplexity_loss, decoded_from_natural

    def save_pretrained(self, output_dir):
        output_dir = Path(output_dir)
        if self.encoder is not None:
            save_model(self.encoder, create_dir(output_dir / "encoder"))
        if self.decoder is not None:
            save_model(self.decoder, create_dir(output_dir / "decoder"))
        # save_tokenizer(self.tokenizer, create_dir(output_dir / "tokenizer"))

        # # Good practice: save your training arguments together with the trained model
        extra_args = {
            "model_name": self.model_name,
            "padding_token_id": self.padding_token_id,
            "uses_perplexity_loss": self.perplexity_calculator is not None,
        }
        torch.save(extra_args, os.path.join(output_dir, "extra_args.json"))
        
    @classmethod
    def from_pretrained(cls, output_dir):
        output_dir = Path(output_dir)
        
        extra_args = torch.load(os.path.join(output_dir, "extra_args.json"))
        model_name = extra_args["model_name"]
        padding_token_id = extra_args["padding_token_id"]
        uses_perplexity_loss = extra_args["uses_perplexity_loss"]
        
        # AutoModel.from_pretrained loads wrong model if it is AutoModelForCausalLM, so we load this class as well
        encoder = AutoModelForCausalLM.from_pretrained(output_dir / "encoder") if (output_dir / "encoder").exists() else None
        decoder = AutoModelForCausalLM.from_pretrained(output_dir / "decoder") if (output_dir / "encoder").exists() else None
        # tokenizer = AutoTokenizer.from_pretrained(output_dir / "tokenizer")
        perplexity_calculator = get_perplexity_calculator(model_name) if uses_perplexity_loss else None
        
        return cls(model_name=model_name, encoder=encoder, decoder=decoder, perplexity_calculator=perplexity_calculator, padding_token_id=padding_token_id)


def get_perplexity_calculator(model_name):
    # uses causal LM
    return PerplexityCalculator(AutoModelForCausalLM.from_pretrained(model_name))


def load_model(model_name, wait_token='<w>', use_pretrained=True, use_perplexity_loss=True, use_encoder=False,
               use_decoder=False):
    """
    Load a model with the option to initialize with pretrained weights or randomly.

    Args:
    model_name (str): The name of the model to load (e.g., 'gpt2', 'llama-2').
    use_pretrained (bool): Whether to load the model with pretrained weights. If False, initializes with random weights.

    Returns:
    tuple: A tuple containing the tokenizer and the model.
    """
    assert (use_decoder or use_encoder)

    if use_perplexity_loss:
        perplexity_calculator = get_perplexity_calculator(model_name)
    else:
        perplexity_calculator = None
    
    if use_pretrained:
        load_model_fn = AutoModelForCausalLM.from_pretrained
    else:
        load_model_fn = functools.partial(
            load_pretrained_config_from_scratch,
            auto_model_class=AutoModelForCausalLM,
        )
        
    if not use_encoder:
        assert not use_perplexity_loss, 'No real use case of using perplexity loss when there is no encoder.'
        encoder = None
    else:
        encoder = load_model_fn(model_name)

    if not use_decoder:
        decoder = None
    else:
        decoder = load_model_fn(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens([wait_token])  # add a special wait token
    wait_id = tokenizer.convert_tokens_to_ids(wait_token)

    if encoder is not None:
        encoder.resize_token_embeddings(len(tokenizer))

    if decoder is not None:
        decoder.resize_token_embeddings(len(tokenizer))

    if perplexity_calculator is not None:
        perplexity_calculator.resize_token_embeddings(len(tokenizer))

    return AutoEncoderLLM(model_name, encoder, decoder, perplexity_calculator, tokenizer.pad_token), tokenizer, wait_id


#%%
def test(*args, **kwargs):
    tmp_dir = tempfile.mkdtemp()
    autoencoder, tokenizer, wait_id = load_model("gpt2", *args, **kwargs)
    # print("Short info1:", autoencoder.short_info())
    autoencoder.save_pretrained(tmp_dir)
    list(autoencoder.parameters())
    autoencoder2 = AutoEncoderLLM.from_pretrained(tmp_dir)
    # print()
    # print("Short info2:", autoencoder2.short_info())
    
    assert autoencoder.same_archs(autoencoder2), f"autoencoder: {autoencoder.short_info()}\nautoencoder2: {autoencoder2.short_info()}"
if __name__ == "__main__":
    for use_pretrained in [True, False]:
        for use_perplexity_loss in [True, False]:
            for use_decoder in [True, False]:
                for use_encoder in [True, False]:
                    if not use_decoder and not use_encoder:
                        continue
                    if use_perplexity_loss and not use_encoder:
                        continue
                    print(f"use_pretrained={use_pretrained}, use_perplexity_loss={use_perplexity_loss}, {use_decoder=}, {use_encoder=}")
                    test(use_pretrained=use_pretrained, use_perplexity_loss=use_perplexity_loss, use_decoder=use_decoder, use_encoder=use_encoder)
                
    # test(use_pretrained=True, use_perplexity_loss=False, decoder_only=True)

# %%
