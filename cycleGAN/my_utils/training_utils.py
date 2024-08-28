#%%
import itertools
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import pandas as pd
import json

from accelerate import PartialState, Accelerator

from model_preparation import AutoEncoderLLM
try:
    from .generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe
except ImportError:
    from my_utils.generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe


class Checkpointer:
    """Save model, but only if it is better than the previous best model"""
    def __init__(self, output_dir, args):
        self.prev_validation_loss = 9999999999
        self.output_dir = output_dir
        self.args_dict = vars(args)

    def checkpoint(self, accelerator, model, validation_loss):
        if accelerator.is_main_process:
            if self.args_dict is not None:
                with open('cmd_args.json', 'w') as json_file:
                    json.dump(self.args_dict, json_file, indent=4)
                self.args_dict = None
                
        unwrapped_model = accelerator.unwrap_model(model)
        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        if validation_loss < self.prev_validation_loss:
            self.prev_validation_loss = validation_loss

            unwrapped_model.save_pretrained(
                self.output_dir,
                is_main_process=accelerator.is_main_process,
                save_function=accelerator.save,
                state_dict=accelerator.get_state_dict(model)
            )
            print(f'Model saved in {self.output_dir}')

def prepare_formal_natural_inputs(formal_texts, natural_texts, tokenizer, return_natural_inputs=True):
    """
    Prepare inputs for the model by tokenizing and converting to tensors on right device
    """
    # Encode formal to natural
    formal_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
    formal_inputs = {k: v.to(PartialState().device) for k, v in
                formal_inputs.items()}  # Ensure inputs are on the right device

    # This has to be deterministic else some process will wait for the others!
    if return_natural_inputs:
        natural_inputs = tokenizer(natural_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        natural_inputs = {k: v.to(PartialState().device) for k, v in
                    natural_inputs.items()}  # Ensure inputs are on the right device
    else:
        natural_inputs = None
        
    return formal_inputs, natural_inputs
        

def create_val_metrics_string(metrics):
    return f'val_rec_l/rf: {metrics["recon_loss"]:.3f}/ {metrics["rf_recon_loss"]:.3f}, ' \
            f'val_enc_l/rf: {metrics["enc_loss"]:.3f}/ {metrics["rf_enc_loss"]:.3f}, ' \
            f'val_perp_l/rf: {metrics["log_perplexity_loss"]:.3f}/ {metrics["rf_log_perplexity_loss"]:.3f}'


def run_validation(
    model: AutoEncoderLLM, accelerator: Accelerator, tokenizer, 
    non_rephrased_dataloader, rephrased_dataloader, 
    df_filename,
    model_kwargs: Optional[Dict] = None, max_num_batches=None,
):
    non_rf_avg_encoder_loss, non_rf_avg_perplexity_loss, non_rf_avg_decoder_loss, non_rf_df = \
        compute_loss_and_df(model, accelerator, tokenizer, data_loader=non_rephrased_dataloader, model_kwargs=model_kwargs, max_num_batches=max_num_batches)
    rf_avg_encoder_loss, rf_avg_perplexity_loss, rf_avg_decoder_loss, rf_df = \
        compute_loss_and_df(model, accelerator, tokenizer, data_loader=rephrased_dataloader, model_kwargs=model_kwargs, max_num_batches=max_num_batches)
        
    metrics = {
        "recon_loss": non_rf_avg_decoder_loss, "enc_loss": non_rf_avg_encoder_loss, "log_perplexity_loss": non_rf_avg_perplexity_loss,
        "rf_recon_loss": rf_avg_decoder_loss, "rf_enc_loss": rf_avg_encoder_loss, "rf_log_perplexity_loss": rf_avg_perplexity_loss,
    }

    if PartialState().is_main_process:    
        marker_row = pd.DataFrame(
            {column: "" if column != 'formal_target'
                else "---- START OF REPHRASED NL DATAFRAME ----" for column in non_rf_df.columns},
            index=[0])

        # Concatenate the first DataFrame, marker row, and the second DataFrame
        combined_df = pd.concat([non_rf_df, marker_row, rf_df], ignore_index=True)

        combined_df.to_csv(df_filename, index=False, encoding='utf-8')
        
    return metrics

def as_dict(**kwargs):
    return kwargs
# def compute_validation(accelerator, ae_model, args, batch_idx, epoch, tokenizer, val_dataloader,
#                        val_rephrased_dataloader, valid_recon_save_path, wait_id, chkpt_bst_mdl_every, checkpointer):
#     """
#     Compute metrics on two data loaders, 
#     save data frames,
#     checkpoint model
#     """
#     # Validation
#     ae_model.eval()
#     val_ae_inputs = {'input_ids': None, 'attention_mask': None}
#     with torch.no_grad():
#         val_enc_loss, val_log_perplexity_loss, val_recon_loss, df = validate_given_data_loader(
#             accelerator, ae_model, args, tokenizer, val_ae_inputs, iter(val_dataloader), wait_id)

#         # rf = rephrased
#         val_rf_enc_loss, val_rf_log_perplexity_loss, val_rf_recon_loss, df_rf = validate_given_data_loader(
#             accelerator, ae_model, args, tokenizer, val_ae_inputs, iter(val_rephrased_dataloader), wait_id)

#         metrics = {
#             "recon_loss": val_recon_loss, "enc_loss": val_enc_loss, "log_perplexity_loss": val_log_perplexity_loss,
#             "rf_recon_loss": val_rf_recon_loss, "rf_enc_loss": val_rf_enc_loss, "rf_log_perplexity_loss": val_rf_log_perplexity_loss
#         }
#         # val_update = f'val_rec_l/rf: {val_recon_loss:.3f}/ {val_rf_recon_loss:.3f}, ' \
#         #              f'val_enc_l/rf: {val_enc_loss:.3f}/ {val_rf_enc_loss:.3f}, ' \
#         #              f'val_perp_l/rf: {val_log_perplexity_loss:.3f}/ {val_rf_log_perplexity_loss:.3f}'

#         # Save the DataFrame to a CSV file
#         # Create a marker DataFrame with one row that indicates the start of the second DataFrame
#         marker_row = pd.DataFrame(
#             {column: "" if column != 'formal_target'
#                 else "---- START OF REPHRASED NL DATAFRAME ----" for column in df.columns},
#             index=[0])

#         # Concatenate the first DataFrame, marker row, and the second DataFrame
#         combined_df = pd.concat([df, marker_row, df_rf], ignore_index=True)

#         file_name = f'{epoch}_{batch_idx}_fl_fl.csv'
#         combined_df.to_csv(os.path.join(valid_recon_save_path, file_name), index=False, encoding='utf-8')

#         ae_model.train()
#         if batch_idx % chkpt_bst_mdl_every == (chkpt_bst_mdl_every - 1):
#             checkpointer.checkpoint(accelerator, ae_model, val_recon_loss + val_enc_loss)

#     return metrics

# import evaluate
# import datasets
# import numpy as np
# class Averager(evaluate.Metric):
#     def _info(self):
#         return evaluate.MetricInfo(
#             description="Averages a list of values",
#             inputs_description="predictions: list of float",
#             citation="Maximilian Mordig",
#             features=datasets.Features(self._get_feature_types()),
#             reference_urls=[
#                 "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html"
#             ],
#         )

#     def _get_feature_types(self):
#         return {
#             "predictions": datasets.Value("float"),
#             "references": datasets.Value("float"),
#         }

#     def _compute(self, predictions, references, weights=None):
#         # return {"avg": np.asarray(predictions).mean()}
#         return {"avg": np.average(predictions, weights=weights)}

# # averager = Averager()
# # averager.add_batch(predictions=np.arange(10), references=np.arange(10))
# # # averager.compute(weights=np.arange(1,11))
# # averager.compute()

# run on last batch without gather
def decode_logits_or_inputs(tokenizer, logits_or_inputs, compress):
    if logits_or_inputs is None:
        return None
    if isinstance(logits_or_inputs, dict):
        if "inputs_embeds" in logits_or_inputs:
            # cannot be decoded
            return None
        token_ids = logits_or_inputs["input_ids"]
    elif logits_or_inputs.ndim == 3:
        token_ids = torch.argmax(logits_or_inputs, dim=-1)
    else:
        # labels are not a dict, but a tensor of shape (batch_size, seq_len)
        token_ids = logits_or_inputs
    assert token_ids.ndim == 2
    text = tokenizer.batch_decode(token_ids.cpu(), skip_special_tokens=False)
    if compress:
        return batch_compress_text_forwaiting_and_eot_tokens(text, eot_token=tokenizer.eos_token)
    return text


from accelerate import Accelerator
def compute_loss_and_df(model: AutoEncoderLLM, accelerator: Accelerator, tokenizer, data_loader, model_kwargs: Optional[Dict] = None, max_num_batches=None):
    """
    Compute loss on given dataloader
    
    This does not handle the situation at the end of the dataloader well (due to duplicated data)
    
    Returns:
        
    """
    if model_kwargs is None:
        model_kwargs = {}
        
    perplexity_losses = None
    encoder_losses = None
    decoder_losses = None
    sizes = None
    
    def combine_handling_none(x, val):
        """combine array x and value, handling x=None"""
        if x is None:
            return np.array([val])
        return np.concatenate([x, [val]])
    
    model.eval()
    with torch.no_grad():
        is_first_batch = True
        for batch in itertools.islice(data_loader, max_num_batches):
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding
            
            formal_inputs, natural_inputs = prepare_formal_natural_inputs(
                formal_texts, natural_texts, tokenizer=tokenizer,
            )

            model_outputs, model_inputs = model(formal_inputs=formal_inputs, natural_inputs=natural_inputs, **model_kwargs, return_inputs=is_first_batch)
            if model_outputs.log_perplexity_loss is not None:
                perplexity_loss = accelerator.gather(model_outputs.log_perplexity_loss).cpu().numpy()
                perplexity_losses = combine_handling_none(perplexity_losses, perplexity_loss)
            if model_outputs.encoder_outputs is not None:
                encoder_loss = accelerator.gather(model_outputs.encoder_outputs.loss).cpu().numpy()
                encoder_losses = combine_handling_none(encoder_losses, encoder_loss)
            if model_outputs.decoder_outputs is not None:
                decoder_loss = accelerator.gather(model_outputs.decoder_outputs.loss).cpu().numpy()
                decoder_losses = combine_handling_none(decoder_losses, decoder_loss)
            # not working as of now
            # ones_for_batch = torch.ones(len(formal_texts), device=formal_inputs["input_ids"].device)
            # sizes = combine_handling_none(sizes, accelerator.gather_for_metrics(ones_for_batch).cpu().numpy())
            
            if is_first_batch:
                encoder_inputs_decoded = decode_logits_or_inputs(tokenizer, model_inputs.encoder_inputs, compress=True)
                encoder_targets_decoded = decode_logits_or_inputs(tokenizer, model_inputs.encoder_targets, compress=True)
                encoder_outputs_decoded = decode_logits_or_inputs(tokenizer, model_outputs.encoder_logits(), compress=True)
                
                decoder_inputs_decoded = decode_logits_or_inputs(tokenizer, model_inputs.decoder_inputs, compress=True)
                decoder_targets_decoded = decode_logits_or_inputs(tokenizer, model_inputs.decoder_targets, compress=True)
                decoder_outputs_decoded = decode_logits_or_inputs(tokenizer, model_outputs.decoder_logits(), compress=True)
                decoder_output_from_natural_decoded = decode_logits_or_inputs(tokenizer, model_outputs.decoder_logits_from_natural(), compress=True)
                
            is_first_batch = False
    
    avg_perplexity_loss = np.average(perplexity_losses, weights=sizes) if perplexity_losses is not None else 0
    avg_encoder_loss = np.average(encoder_losses, weights=sizes) if encoder_losses is not None else 0
    avg_decoder_loss = np.average(decoder_losses, weights=sizes) if decoder_losses is not None else 0
    
    df = make_pandas_dataframe(**{
        'encoder_input': encoder_inputs_decoded,
        'encoder_target': encoder_targets_decoded,
        'encoder_output': encoder_outputs_decoded,
        
        'decoder_input': decoder_inputs_decoded,
        'decoder_target': decoder_targets_decoded,
        'decoder_output': decoder_outputs_decoded,
        'decoder_output_from_natural': decoder_output_from_natural_decoded,
        
        'avg_encoder_loss': avg_encoder_loss,
        'avg_perplexity_loss': avg_perplexity_loss,
        'avg_decoder_loss': avg_decoder_loss
    })

    return avg_encoder_loss, avg_perplexity_loss, avg_decoder_loss, df
        
    

# def validate_given_data_loader_old(ae_model: AutoEncoderLLM, accelerator: Accelerator, tokenizer, data_loader, model_kwargs: Optional[Dict] = None, max_num_batches=None):
#     """
#     Compute loss on given dataloader
#     """
#     val_recon_loss = 0
#     val_enc_loss = 0
#     val_log_perplexity_loss = val_log_perplexity_loss_batch = 0
#     enc_texts = recon_texts = val_formal_texts_2_save = None
#     val_rec_outputs = val_enc_outputs = val_encoder_target = val_ae_target = decoded_frm_nl_texts = None
#     num_batches = 0
#     for batch in itertools.islice(data_loader, max_num_batches):
#         num_batches += 1

#         formal_texts = batch['formal']
#         natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding
        
#         formal_inputs, natural_inputs = prepare_formal_natural_inputs(
#             formal_texts, natural_texts, tokenizer=tokenizer,
#         )

#         model_outputs = ae_model(formal_inputs=formal_inputs, natural_inputs=natural_inputs, **model_kwargs)
        
#         # val_formal_texts = batch['formal']
#         # val_natural_texts = batch['natural']

#         # enc_label = tokenizer(val_natural_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#         # enc_label = {k: v.to(accelerator.device) for k, v in
#         #              enc_label.items()}  # Ensure inputs are on the right device

#         # # Encode formal to natural and then back
#         # val_enc_inputs = tokenizer(val_formal_texts, return_tensors='pt', padding=True, truncation=True,
#         #                            max_length=512)
#         # val_enc_inputs = {k: v.to(accelerator.device) for k, v in val_enc_inputs.items()}

#         # # perform padding as needed to the inputs and targets for validation data
#         # val_ae_inputs, val_encoder_target, val_ae_target = prepare_inputs(
#         #     encoder_only=(args.use_encoder and not args.use_decoder),
#         #     decoder_only=(args.use_decoder and not args.use_encoder), formal_lang=val_enc_inputs,
#         #     natural_lang=enc_label, wait_id=wait_id, pad_token_id=tokenizer.pad_token_id)

#         # dec_natural, _ = \
#         #     introduce_waiting_tokens(enc_label, enc_label, wait_id, pad_token_id=tokenizer.pad_token_id)

#         # val_enc_outputs, val_rec_outputs, val_log_perplexity_loss_batch, decoded_from_natural = \
#         #     ae_model(**val_ae_inputs, output_hidden_states=True, recon_target=val_ae_target,
#         #              encoder_target=val_encoder_target, decode_natural=dec_natural)

#         decoded_from_natural = model_outputs.decoder_outputs_from_natural
#         if decoded_from_natural is not None:
#             decoded_token_ids = torch.argmax(decoded_from_natural.logits, dim=-1)
#             decoded_frm_nl_texts = tokenizer.batch_decode(decoded_token_ids, skip_special_tokens=False)

#         if val_rec_outputs is not None:
#             val_recon_loss += val_rec_outputs.loss
#         if val_enc_outputs is not None:
#             val_enc_loss += val_enc_outputs.loss
#         val_log_perplexity_loss += val_log_perplexity_loss_batch

#     # print(get_process_cuda_memory_info())
#     # Convert logits to predicted token IDs
#     val_enc_loss /= num_batches
#     val_log_perplexity_loss /= num_batches
#     val_recon_loss /= num_batches

#     if val_rec_outputs is not None:
#         recon_token_ids = torch.argmax(val_rec_outputs.logits, dim=-1)
#         # Convert token IDs to text
#         recon_texts = tokenizer.batch_decode(recon_token_ids, skip_special_tokens=False)
#         val_formal_texts_2_save = tokenizer.batch_decode(val_ae_target, skip_special_tokens=False)

#     # Get encoded text
#     if val_enc_outputs is not None:
#         enc_tok_ids = torch.argmax(val_enc_outputs.logits, dim=-1)
#         enc_texts = tokenizer.batch_decode(enc_tok_ids, skip_special_tokens=False)

#         val_formal_texts_2_save = tokenizer.batch_decode(val_ae_inputs['input_ids'], skip_special_tokens=False)

#     if isinstance(val_encoder_target, dict):
#         val_encoder_target = val_encoder_target['input_ids']
#     val_natural_target = tokenizer.batch_decode(val_encoder_target, skip_special_tokens=False)
#     df = make_pandas_dataframe(**{
#         'formal_target/(input_if_encoder_only)': batch_compress_text_forwaiting_and_eot_tokens(
#             val_formal_texts_2_save, eot_token=tokenizer.eos_token),
#         'formal_reconstructed (if decoder only: same as formal_generated)':
#             batch_compress_text_forwaiting_and_eot_tokens(recon_texts, eot_token=tokenizer.eos_token),
#         'natural_created': batch_compress_text_forwaiting_and_eot_tokens(enc_texts,
#                                                                          eot_token=tokenizer.eos_token),
#         'natural_target': batch_compress_text_forwaiting_and_eot_tokens(val_natural_target,
#                                                                         eot_token=tokenizer.eos_token),
#         'formal_generated (if decoder only: same as formal_reconstructed)':
#             batch_compress_text_forwaiting_and_eot_tokens(decoded_frm_nl_texts, eot_token=tokenizer.eos_token),
#         'average_val_enc_loss': val_enc_loss,
#         'average_val_log_perplexity_loss': val_log_perplexity_loss,
#         'average_val_recon_loss': val_recon_loss
#     })

#     return val_enc_loss, val_log_perplexity_loss, val_recon_loss, df


# # todo: remove
# def introduce_waiting_tokens_old(inputs, targets, wait_token_id, padding_id):
#     """
#     Introduce waiting tokens so that targets only starts once all inputs have been consumed.
    
#     It inserts wait tokens at the beginning of the target sequence, and padding tokens at the end.
#     It inserts padding tokens at the end of the input sequence.
#     Finally, it returns the new input (input_ids, attention_mask) and target (input_ids).
    
#     It assumes right padding.
    
#     Arguments:
#         inputs: tokenized natural texts, of shape (batch_size, seq_len)
#         targets: tokenized formal texts, of shape (batch_size, seq_len)
#     """
#     target_ids = []
#     padded_inp_ids = []
#     padded_inp_masks = []
#     for data_id in range(inputs['attention_mask'].shape[0]):
#         n_wait = inputs['attention_mask'][data_id].tolist().count(1)
#         n_pad = inputs['attention_mask'][data_id].shape[0] - n_wait
#         # Create a list of waiting tokens
#         wait_tokens = [wait_token_id] * n_wait  # ensure that n_wait is the length of inputs or a fixed size
#         target_ids.append(torch.tensor(wait_tokens + targets['input_ids'][data_id].tolist() +
#                                            targets['input_ids'][data_id][-1:].tolist() * n_pad, dtype=torch.long,
#                                        device=inputs['input_ids'].device))
#         padding_len = target_ids[-1].shape[0] - inputs['input_ids'][data_id].shape[0]
#         padded_inp_ids.append(inputs['input_ids'][data_id].tolist() + [padding_id] * padding_len)
#         # one end of text or end of sentence token must be included.
#         padded_inp_masks.append(inputs['attention_mask'][data_id].tolist() + [1] + [0] * (padding_len - 1))
#     return {'input_ids': torch.tensor(padded_inp_ids, dtype=torch.long, device=inputs['input_ids'].device),
#             'attention_mask': torch.tensor(padded_inp_masks, dtype=torch.long, device=inputs['input_ids'].device)}, \
#         torch.stack(target_ids, dim=0)
        

# def introduce_waiting_tokens_for_encoderdecoder(formal_inputs, natural_inputs, wait_id, padding_id):
#     """
#     Handle encoder and decoder
    
#     Arguments:
#         formal_inputs: tokenized formal texts, of shape (batch_size, seq_len)
#         natural_inputs: tokenized natural texts, of shape (batch_size, seq_len)
#     """
    
#     # encoder_inputs, decoder_inputs refers to inputs including labels
#     encoder_inputs, encoder_labels = introduce_waiting_tokens(formal_inputs, formal_inputs, wait_token_id=wait_token_id, padding_id=padding_id)
#     if natural_inputs is None:
#         return (encoder_inputs, encoder_labels), (None, None)
    
#     decoder_inputs, decoder_labels = introduce_waiting_tokens(encoder_inputs, natural_inputs, wait_token_id=wait_id, padding_id=padding_id)
#     return (encoder_inputs, encoder_labels), (decoder_inputs, decoder_labels)

# def introduce_waiting_tokens_for_ae_old(enc_inputs, enc_label, wait_id, padding_id):
#     """
#     introduce waiting tokens for the encoder-decoder architecture
    
#     Arguments:
#         enc_inputs: tokenized formal texts, of shape (batch_size, seq_len)
#         enc_label: tokenized natural texts, of shape (batch_size, seq_len)
#     """
#     enc_inputs_paded_as_recon_target, recon_target = \
#         introduce_waiting_tokens(enc_inputs, enc_inputs, wait_id, padding_id)

#     if enc_label is None:
#         # no NL label
#         return enc_inputs_paded_as_recon_target, None, recon_target

#     enc_inputs_paded_as_enc_target, enc_target = introduce_waiting_tokens(enc_inputs, enc_label, wait_id, padding_id)

#     if enc_target.shape[-1] > recon_target.shape[-1]:
#         recon_target = F.pad(recon_target, (0, enc_target.shape[-1] - recon_target.shape[-1]), value=padding_id)
#         return enc_inputs_paded_as_enc_target, enc_target, recon_target
#     elif enc_target.shape[-1] < recon_target.shape[-1]:
#         enc_target = F.pad(enc_target, (0, recon_target.shape[-1] - enc_target.shape[-1]), value=padding_id)
#         return enc_inputs_paded_as_recon_target, enc_target, recon_target
#     else:
#         return enc_inputs_paded_as_enc_target, enc_target, recon_target


# def prepare_inputs(encoder_only, decoder_only, formal_lang, natural_lang, wait_id, pad_token_id):
#     """
#     ae_target: decoder_target
#     """
#     if decoder_only:
#         # only needs the encoder lab
#         assert natural_lang is not None, f'For decoder we need a natural lang target'
#         encoder_objs = {'input_ids': None, 'attention_mask': None}
#         decoder_objs = introduce_waiting_tokens(natural_lang, formal_lang, wait_id,
#                                                              padding_id=pad_token_id)
#     elif encoder_only:
#         assert natural_lang is not None, f'For encoder we need a natural lang inputs'
#         encoder_objs = introduce_waiting_tokens(formal_lang, natural_lang, wait_id,
#                                                              padding_id=pad_token_id)
#         decoder_objs = {'input_ids': None, 'attention_mask': None}
#     else:
#         encoder_objs, decoder_objs = introduce_waiting_tokens_for_encoderdecoder(
#             formal_lang, natural_lang, wait_id, padding_id=pad_token_id)

#     return encoder_objs, decoder_objs



        
#%%

# ",".join([f"{x:2d}" for x in xx.tolist()])
# #%%
#     enc_in, enc_targets, rec_target = introduce_waiting_tokens_for_encoderdecoder(inputs, targets, -1, 0)
#     print(enc_in)
#     print(enc_targets)
#     print(rec_target)
#     print()

# #%%
#     # inputs of length 6, targets of length 4
#     inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6],
#                                          [1, 2, 0, 0, 0, 0]], dtype=torch.long),
#               'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1],
#                                               [1, 1, 0, 0, 0, 0]], dtype=torch.long)}
#     targets = {'input_ids': torch.tensor([[11, 12, 13, 14],
#                                           [11, 12, 13, 0, ]], dtype=torch.long),
#                'attention_mask': torch.tensor([[1, 1, 1, 1],
#                                                [1, 1, 1, 0]], dtype=torch.long)}
#     # inputs, targets = introduce_waiting_tokens(inputs, inputs, -1, 0)
#     enc_in, enc_targets, rec_target = introduce_waiting_tokens_for_ae(inputs, targets, -1, 0)
#     print(enc_in)
#     print(enc_targets)
#     print(rec_target)

# %%
