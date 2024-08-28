#%%
from dataclasses import dataclass
import functools
import os
from pathlib import Path
import tempfile
from typing import Optional, Tuple
import torch
from utils import create_dir, is_frozen, load_pretrained_config_from_scratch, save_model, save_tokenizer
from frozen_discriminator import PerplexityCalculator
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer, AutoModelForCausalLM, AutoModel
import textwrap
from transformers.utils import ModelOutput

@dataclass
class AutoEncoderLLMOutput(ModelOutput):
    """Output of the AutoEncoderLLM model"""
    encoder_outputs: Optional[ModelOutput]
    decoder_outputs: Optional[ModelOutput] # outputs of decoder, possibly after feeding in encoder outputs
    decoder_outputs_from_natural: Optional[ModelOutput] # outputs of decoder, using ground-truth natural inputs, not set for decoder-only model
    log_perplexity_loss: Optional[torch.Tensor] = None # perplexity loss of encoder output
    
    def __post_init__(self):
        if self.log_perplexity_loss is not None:
            assert self.encoder_outputs is not None, "cannot compute perplexity loss without encoder outputs"
        if self.decoder_outputs_from_natural is not None:
            assert self.encoder_outputs is not None, "should also have computed decoder outputs from formal"
            
    def total_loss(self, enc_loss_weight):
        loss = 0
        if self.encoder_outputs is not None:
            loss += enc_loss_weight * self.encoder_outputs.loss
        if self.decoder_outputs is not None:
            loss += self.decoder_outputs.loss
        if self.log_perplexity_loss is not None:
            loss += self.log_perplexity_loss
        return loss
    
    def encoder_loss(self):
        return self.encoder_outputs.loss if self.encoder_outputs is not None else 0
    
    def decoder_loss(self):
        return self.decoder_outputs.loss if self.decoder_outputs is not None else 0
    
    def encoder_logits(self):
        return self.encoder_outputs.logits if self.encoder_outputs is not None else None
    
    def decoder_logits(self):
        return self.decoder_outputs.logits if self.decoder_outputs is not None else None
    
    def decoder_logits_from_natural(self):
        return self.decoder_outputs_from_natural.logits if self.decoder_outputs_from_natural is not None else None
    
@dataclass
class AutoEncoderLLMInputs:
    """Inputs to the models in the AutoEncoderLLM class"""
    encoder_inputs: Optional[dict]
    encoder_targets: Optional[dict]
    decoder_inputs: Optional[dict]
    decoder_targets: Optional[dict]
    
def introduce_waiting_tokens(inputs, labels, wait_token_id, pad_token_id):
    """
    Introduce wait tokens in labels, so they only start once all inputs have been consumed, pad inputs accordingly.
    
    It inserts wait tokens at the beginning of the label sequence.
    It inserts padding tokens at the end of the input sequence.
    Finally, it returns the new input 
    
    It assumes right padding.
    
    Arguments:
        inputs: tokenized texts, of shape (batch_size, seq_len), or embeddings of shape (batch_size, seq_len, hidden_size)
        labels: tokenized texts, of shape (batch_size, seq_len)
        wait_token_id: token id for the wait token that is prepended to the labels, do not use -100 because the model should not predict the output there already
        padding_id: token id for the padding token for inputs and labels
    
    Returns:
        inputs dict (with keys input_ids, attention_mask) and labels (input_ids).
    """
    
    input_ids_or_embds = inputs["input_ids"] if "input_ids" in inputs else inputs["inputs_embeds"]
    inputs_attention_mask = inputs["attention_mask"]
    label_input_ids, label_attention_mask = labels["input_ids"], labels["attention_mask"]
    bs = input_ids_or_embds.shape[0]
    device = input_ids_or_embds.device
    
    inputs_lens = inputs_attention_mask.sum(dim=1)
    label_lens = label_attention_mask.sum(dim=1)
    total_lens = inputs_lens + label_lens
    
    new_len = total_lens.max()
    if input_ids_or_embds.ndim == 3:
        # set to zeros
        new_input_ids_or_embds = torch.full((bs, new_len, input_ids_or_embds.shape[2]), 0, dtype=input_ids_or_embds.dtype, device=device)
    else:
        new_input_ids_or_embds = torch.full((bs, new_len), pad_token_id, dtype=torch.long, device=device)
    new_attention_mask = torch.zeros((bs, new_len), dtype=torch.long, device=device)
    labels = torch.full((bs, new_len), pad_token_id, dtype=torch.long, device=device)
    
    for i in range(bs):
        new_input_ids_or_embds[i, :inputs_lens[i]] = input_ids_or_embds[i, :inputs_lens[i]]
        new_attention_mask[i, :total_lens[i]] = 1
        labels[i, inputs_lens[i]:total_lens[i]] = label_input_ids[i, :label_lens[i]]
        labels[i, :inputs_lens[i]] = wait_token_id
        
    return {
        "inputs_embeds" if input_ids_or_embds.ndim == 3 else "input_ids": new_input_ids_or_embds,
        "attention_mask": new_attention_mask,
        # "labels": labels,
    }, labels
    
@functools.lru_cache(maxsize=None)
def print_once(*args, **kwargs):
    print(*args, **kwargs)
    
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
        
        assert encoder is not None or decoder is not None, "At least one of encoder or decoder should be present"
        
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

    def forward(self, formal_inputs, natural_inputs, *, also_decode_natural=False, return_inputs=False, **kwargs) -> Tuple[AutoEncoderLLMOutput, AutoEncoderLLMInputs]:
        """
        Forward through the model
        
        Encoder: formal to natural, decoder: natural to formal
        
        Encoder-only: add waiting tokens to the natural inputs, pad the formal inputs
        Decoder-only: pad the natural inputs, add waiting tokens to the formal inputs
        Encoder and decoder: same as encoder-only, then take encoder outputs, add waiting tokens to the formal, pad encoder outputs
        
        Arguments:
            formal_inputs: dict with keys: input_ids, attention_mask
            natural_inputs: dict with keys: input_ids, attention_mask, may be None
            also_decode_natural: whether to decode the natural language as well, when using encoder and decoder; same as when using decoder-only
            kwargs: extra args to pass to introduce_waiting_tokens
        """
        assert formal_inputs is not None, "formal_inputs should not be None"
        decoder_outputs_from_natural = None
        perplexity_loss = None
        
        encoder_inputs, encoder_targets = None, None
        decoder_inputs, decoder_targets = None, None
        
        if self.encoder is None:
            print_once("Decoder-only forward")
            # decoder-only
            assert natural_inputs is not None, "natural_inputs should not be None"
            encoder_outputs = None
            decoder_inputs, decoder_targets = introduce_waiting_tokens(natural_inputs, formal_inputs, **kwargs)
            decoder_outputs = self.decoder(**decoder_inputs, labels=decoder_targets)
        elif self.decoder is None:
            # encoder-only
            print_once("Encoder-only forward")
            assert natural_inputs is not None, "natural_inputs should not be None"
            encoder_inputs, encoder_targets = introduce_waiting_tokens(formal_inputs, natural_inputs, **kwargs)
            encoder_outputs = self.encoder(**encoder_inputs, labels=encoder_targets)
            decoder_outputs = None
        else:
            # encoder and decoder
            print_once("Encoder-decoder forward")
            if natural_inputs is None:
                encoder_inputs, _ = introduce_waiting_tokens(formal_inputs, formal_inputs, **kwargs)
                encoder_targets = None
            else:
                encoder_inputs, encoder_targets = introduce_waiting_tokens(formal_inputs, natural_inputs, **kwargs)
            encoder_outputs = self.encoder(**encoder_inputs, labels=encoder_targets, output_hidden_states=True)
            
            unpadded_decoder_inputs = encoder_outputs.hidden_states[-1] # shape: (batch_size, seq_len, hidden_size)
            assert unpadded_decoder_inputs.ndim == 3, f"got shape {unpadded_decoder_inputs.shape}"
            decoder_inputs, decoder_targets = introduce_waiting_tokens({
                "inputs_embeds": unpadded_decoder_inputs,
                "attention_mask": encoder_inputs["attention_mask"],
            }, formal_inputs, **kwargs)
            decoder_outputs = self.decoder(**decoder_inputs, labels=decoder_targets)
            
            if also_decode_natural:
                decoder_inputs, decoder_targets = introduce_waiting_tokens(natural_inputs, formal_inputs, **kwargs)
                decoder_outputs_from_natural = self._decode(**natural_inputs, labels=decoder_targets)
            
        if self.perplexity_calculator is not None:
            perplexity_loss = self.perplexity_calculator(inputs_embeds=encoder_outputs.logits)
            
        return AutoEncoderLLMOutput(
            encoder_outputs=encoder_outputs, 
            decoder_outputs=decoder_outputs, 
            decoder_outputs_from_natural=decoder_outputs_from_natural,
            log_perplexity_loss=perplexity_loss,
        ), AutoEncoderLLMInputs(encoder_inputs, encoder_targets, decoder_inputs, decoder_targets) if return_inputs else None
            
    # # todo: remove
    # def forward_old(self, recon_target, encoder_target, decode_natural=None, **enc_inputs):
    #     """
    #     enc_input: Serves as the input to the Autoencode. In ususal operation mode i.e., when bot encoder and the
    #     decoder are present this is a formal text. This goes into encoder that transforms it into natural language and
    #     then teh decoder turn it into a formal language.
    #     In case of decoder only operation, the encoder_target is assumed to be the natural language input
    #     """
    #     # import ipdb; ipdb.set_trace()
    #     if self.encoder is None:  # decoder only model
    #         encoder_outputs = None
    #         decoder_outputs = self._decode(**encoder_target, labels=recon_target)
    #     elif self.decoder is None:  # encoder only model
    #         decoder_outputs = None
    #         encoder_outputs = self._encode(**enc_inputs, labels=encoder_target)
    #     else:  # autoencoder model
    #         encoder_outputs = self._encode(**enc_inputs, labels=encoder_target)
    #         decoder_outputs = self._decode(inputs_embeds=encoder_outputs.hidden_states[-1], labels=recon_target)

    #     if self.perplexity_calculator is None:
    #         log_perplexity_loss = 0
    #     else:
    #         # self.perplexity_calculator.eval()  # should be but deepspeed complains!
    #         # TODO: Also remove the leading <w> tokens? because perplexity on those tokens are weird to compute?
    #         log_perplexity_loss = self.perplexity_calculator(encoder_outputs.logits)

    #     if decode_natural is not None and self.decoder is not None:
    #         decoded_from_natural = self._decode(**decode_natural)
    #     else:
    #         decoded_from_natural = None
    #     return encoder_outputs, decoder_outputs, log_perplexity_loss, decoded_from_natural

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
        decoder = AutoModelForCausalLM.from_pretrained(output_dir / "decoder")
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
        assert not use_perplexity_loss, 'No real use case of using perplexity loss when there is no encoder.' # todo: could compute it perplexity on formal language
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

if __name__ == '__main__':
    
    def format_tensor_helper(t, width=2):
        """print tensor with width 2 per element"""
        # format_tensor_helper( torch.tensor([1, 2, 3, 10]))
        
        # return ",".join([f"{x:2d}" for x in t.tolist()])
        # width
        return ", ".join([f"{x:>{width}}" for x in t.tolist()])

    # inputs of length 6, targets of length 7
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6],
                                         [1, 2, 0, 0, 0, 0]], dtype=torch.long),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1],
                                              [1, 1, 0, 0, 0, 0]], dtype=torch.long)}
    targets = {'input_ids': torch.tensor([[11, 12, 13, 14, 15, 16, 17],
                                          [11, 12, 13, 0, 0, 0, 0]], dtype=torch.long),
               'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                               [1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)}
    # inputs, targets = introduce_waiting_tokens(inputs, inputs, -1, 0)
    
    new_inputs, new_targets = introduce_waiting_tokens(inputs, targets, wait_token_id=-1, pad_token_id=0)
    for i in range(new_inputs["input_ids"].shape[0]):
        print(f"i={i}")
        print("Inputs:        ", format_tensor_helper(new_inputs["input_ids"][i]))
        print("Attention mask:", format_tensor_helper(new_inputs["attention_mask"][i]))
        print("Targets:       ", format_tensor_helper(new_targets[i]))
# %%
