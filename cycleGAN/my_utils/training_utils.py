#%%
import torch
import pandas as pd
import os
import torch.nn.functional as F
try:
    from .generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe
except ImportError:
    from my_utils.generic_utils import batch_compress_text_forwaiting_and_eot_tokens, make_pandas_dataframe


class Checkpointer:
    """Save model, but only if it is better than the previous best model"""
    def __init__(self, output_dir):
        self.prev_validation_loss = 9999999999
        self.output_dir = output_dir

    def checkpoint(self, accelerator, model, validation_loss):
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


def create_metrics_string(metrics):
    return f'val_rec_l/rf: {metrics["recon_loss"]:.3f}/ {metrics["rf_recon_loss"]:.3f}, ' \
            f'val_enc_l/rf: {metrics["enc_loss"]:.3f}/ {metrics["rf_enc_loss"]:.3f}, ' \
            f'val_perp_l/rf: {metrics["log_perplexity_loss"]:.3f}/ {metrics["rf_log_perplexity_loss"]:.3f}'


def compute_validation(accelerator, ae_model, args, batch_idx, epoch, tokenizer, val_dataloader,
                       val_rephrased_dataloader, valid_recon_save_path, wait_id, chkpt_bst_mdl_every, checkpointer):
    """
    Compute metrics on two data loaders, 
    save data frames,
    checkpoint model
    """
    # Validation
    ae_model.eval()
    val_ae_inputs = {'input_ids': None, 'attention_mask': None}
    with torch.no_grad():
        val_enc_loss, val_log_perplexity_loss, val_recon_loss, df = validate_given_data_loader(
            accelerator, ae_model, args, tokenizer, val_ae_inputs, iter(val_dataloader), wait_id)

        # rf = rephrased
        val_rf_enc_loss, val_rf_log_perplexity_loss, val_rf_recon_loss, df_rf = validate_given_data_loader(
            accelerator, ae_model, args, tokenizer, val_ae_inputs, iter(val_rephrased_dataloader), wait_id)

        metrics = {
            "recon_loss": val_recon_loss, "enc_loss": val_enc_loss, "log_perplexity_loss": val_log_perplexity_loss,
            "rf_recon_loss": val_rf_recon_loss, "rf_enc_loss": val_rf_enc_loss, "rf_log_perplexity_loss": val_rf_log_perplexity_loss
        }
        # val_update = f'val_rec_l/rf: {val_recon_loss:.3f}/ {val_rf_recon_loss:.3f}, ' \
        #              f'val_enc_l/rf: {val_enc_loss:.3f}/ {val_rf_enc_loss:.3f}, ' \
        #              f'val_perp_l/rf: {val_log_perplexity_loss:.3f}/ {val_rf_log_perplexity_loss:.3f}'

        # Save the DataFrame to a CSV file
        # Create a marker DataFrame with one row that indicates the start of the second DataFrame
        marker_row = pd.DataFrame(
            {column: "" if column != 'formal_target'
                else "---- START OF REPHRASED NL DATAFRAME ----" for column in df.columns},
            index=[0])

        # Concatenate the first DataFrame, marker row, and the second DataFrame
        combined_df = pd.concat([df, marker_row, df_rf], ignore_index=True)

        file_name = f'{epoch}_{batch_idx}_fl_fl.csv'
        combined_df.to_csv(os.path.join(valid_recon_save_path, file_name), index=False, encoding='utf-8')

        ae_model.train()
        if batch_idx % chkpt_bst_mdl_every == (chkpt_bst_mdl_every - 1):
            checkpointer.checkpoint(accelerator, ae_model, val_recon_loss + val_enc_loss)

    return metrics


def validate_given_data_loader(accelerator, ae_model, args, tokenizer, val_ae_inputs, valid_iterator, wait_id):
    """Compute validation loss"""
    val_recon_loss = 0
    val_enc_loss = 0
    val_log_perplexity_loss = val_log_perplexity_loss_batch = 0
    enc_texts = recon_texts = val_formal_texts_2_save = None
    val_rec_outputs = val_enc_outputs = val_encoder_target = val_ae_target = decoded_frm_nl_texts = None
    for i, val_batch in enumerate(valid_iterator):
        if i > args.valid_for_batches:
            break
        val_formal_texts = val_batch['formal']
        val_natural_texts = val_batch['natural']

        enc_label = tokenizer(val_natural_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        enc_label = {k: v.to(accelerator.device) for k, v in
                     enc_label.items()}  # Ensure inputs are on the right device

        # Encode formal to natural and then back
        val_enc_inputs = tokenizer(val_formal_texts, return_tensors='pt', padding=True, truncation=True,
                                   max_length=512)
        val_enc_inputs = {k: v.to(accelerator.device) for k, v in val_enc_inputs.items()}

        # perform padding as needed to the inputs and targets for validation data
        val_ae_inputs, val_encoder_target, val_ae_target = prepare_inputs(
            encoder_only=(args.use_encoder and not args.use_decoder),
            decoder_only=(args.use_decoder and not args.use_encoder), formal_lang=val_enc_inputs,
            natural_lang=enc_label, wait_id=wait_id, pad_token_id=tokenizer.pad_token_id)

        dec_natural, _ = \
            introduce_waiting_tokens(enc_label, enc_label, wait_id, padding_id=tokenizer.pad_token_id)

        val_enc_outputs, val_rec_outputs, val_log_perplexity_loss_batch, decoded_from_natural = \
            ae_model(**val_ae_inputs, output_hidden_states=True, recon_target=val_ae_target,
                     encoder_target=val_encoder_target, decode_natural=dec_natural)

        if decoded_from_natural is not None:
            decoded_token_ids = torch.argmax(decoded_from_natural.logits, dim=-1)
            decoded_frm_nl_texts = tokenizer.batch_decode(decoded_token_ids, skip_special_tokens=False)

        if val_rec_outputs is not None:
            val_recon_loss += val_rec_outputs.loss
        if val_enc_outputs is not None:
            val_enc_loss += val_enc_outputs.loss
        val_log_perplexity_loss += val_log_perplexity_loss_batch

    # print(get_process_cuda_memory_info())
    # Convert logits to predicted token IDs
    val_enc_loss /= i
    val_log_perplexity_loss /= i
    val_recon_loss /= i

    if val_rec_outputs is not None:
        recon_token_ids = torch.argmax(val_rec_outputs.logits, dim=-1)
        # Convert token IDs to text
        recon_texts = tokenizer.batch_decode(recon_token_ids, skip_special_tokens=False)
        val_formal_texts_2_save = tokenizer.batch_decode(val_ae_target, skip_special_tokens=False)

    # Get encoded text
    if val_enc_outputs is not None:
        enc_tok_ids = torch.argmax(val_enc_outputs.logits, dim=-1)
        enc_texts = tokenizer.batch_decode(enc_tok_ids, skip_special_tokens=False)

        val_formal_texts_2_save = tokenizer.batch_decode(val_ae_inputs['input_ids'], skip_special_tokens=False)

    if isinstance(val_encoder_target, dict):
        val_encoder_target = val_encoder_target['input_ids']
    val_natural_target = tokenizer.batch_decode(val_encoder_target, skip_special_tokens=False)
    df = make_pandas_dataframe(**{
        'formal_target/(input_if_encoder_only)': batch_compress_text_forwaiting_and_eot_tokens(
            val_formal_texts_2_save, eot_token=tokenizer.eos_token),
        'formal_reconstructed (if decoder only: same as formal_generated)':
            batch_compress_text_forwaiting_and_eot_tokens(recon_texts, eot_token=tokenizer.eos_token),
        'natural_created': batch_compress_text_forwaiting_and_eot_tokens(enc_texts,
                                                                         eot_token=tokenizer.eos_token),
        'natural_target': batch_compress_text_forwaiting_and_eot_tokens(val_natural_target,
                                                                        eot_token=tokenizer.eos_token),
        'formal_generated (if decoder only: same as formal_reconstructed)':
            batch_compress_text_forwaiting_and_eot_tokens(decoded_frm_nl_texts, eot_token=tokenizer.eos_token),
        'average_val_enc_loss': val_enc_loss,
        'average_val_log_perplexity_loss': val_log_perplexity_loss,
        'average_val_recon_loss': val_recon_loss
    })

    return val_enc_loss, val_log_perplexity_loss, val_recon_loss, df


def introduce_waiting_tokens(inputs, targets, wait_token_id, padding_id):
    """
    Introduce waiting tokens for the decoder-only architecture
    Given inputs and targets, shift the targets by wait tokens so they effectively start
    once all the input is seen.
    
    Arguments:
        inputs: tokenized natural texts, of shape (batch_size, seq_len)
        targets: tokenized formal texts, of shape (batch_size, seq_len)
    """
    target_ids = []
    padded_inp_ids = []
    padded_inp_masks = []
    for data_id in range(inputs['attention_mask'].shape[0]):
        n_wait = inputs['attention_mask'][data_id].tolist().count(1)
        n_pad = inputs['attention_mask'][data_id].shape[0] - n_wait
        # Create a list of waiting tokens
        wait_tokens = [wait_token_id] * n_wait  # ensure that n_wait is the length of inputs or a fixed size
        target_ids.append(torch.tensor(wait_tokens + targets['input_ids'][data_id].tolist() +
                                           targets['input_ids'][data_id][-1:].tolist() * n_pad, dtype=torch.long,
                                       device=inputs['input_ids'].device))
        padding_len = target_ids[-1].shape[0] - inputs['input_ids'][data_id].shape[0]
        padded_inp_ids.append(inputs['input_ids'][data_id].tolist() + [padding_id] * padding_len)
        # one end of text or end of sentence token must be included.
        padded_inp_masks.append(inputs['attention_mask'][data_id].tolist() + [1] + [0] * (padding_len - 1))
    return {'input_ids': torch.tensor(padded_inp_ids, dtype=torch.long, device=inputs['input_ids'].device),
            'attention_mask': torch.tensor(padded_inp_masks, dtype=torch.long, device=inputs['input_ids'].device)}, \
        torch.stack(target_ids, dim=0)


def introduce_waiting_tokens_for_ae(enc_inputs, enc_label, wait_id, padding_id):
    """
    introduce waiting tokens for the encoder-decoder architecture
    
    Arguments:
        enc_inputs: tokenized formal texts, of shape (batch_size, seq_len)
        enc_label: tokenized natural texts, of shape (batch_size, seq_len)
    """
    enc_inputs_paded_as_recon_target, recon_target = \
        introduce_waiting_tokens(enc_inputs, enc_inputs, wait_id, padding_id)

    if enc_label is None:
        return enc_inputs_paded_as_recon_target, None, recon_target

    enc_inputs_paded_as_enc_target, enc_target = introduce_waiting_tokens(enc_inputs, enc_label, wait_id, padding_id)

    if enc_target.shape[-1] > recon_target.shape[-1]:
        recon_target = F.pad(recon_target, (0, enc_target.shape[-1] - recon_target.shape[-1]), value=padding_id)
        return enc_inputs_paded_as_enc_target, enc_target, recon_target
    elif enc_target.shape[-1] < recon_target.shape[-1]:
        enc_target = F.pad(enc_target, (0, recon_target.shape[-1] - enc_target.shape[-1]), value=padding_id)
        return enc_inputs_paded_as_recon_target, enc_target, recon_target
    else:
        return enc_inputs_paded_as_enc_target, enc_target, recon_target


def prepare_inputs(encoder_only, decoder_only, formal_lang, natural_lang, wait_id, pad_token_id):
    if decoder_only:
        assert natural_lang is not None, f'For decoder we need a natural lang target'
        encoder_target, ae_target = introduce_waiting_tokens(natural_lang, formal_lang, wait_id,
                                                             padding_id=pad_token_id)
        ae_inputs = {'input_ids': None, 'attention_mask': None}
    elif encoder_only:
        assert natural_lang is not None, f'For encoder we need a natural lang inputs'
        ae_inputs, encoder_target = introduce_waiting_tokens(formal_lang, natural_lang, wait_id,
                                                             padding_id=pad_token_id)
        ae_target = None
    else:
        ae_inputs, encoder_target, ae_target = introduce_waiting_tokens_for_ae(
            formal_lang, natural_lang, wait_id, padding_id=pad_token_id)

    return ae_inputs, encoder_target, ae_target


if __name__ == '__main__':
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
    
    enc_in, enc_targets = introduce_waiting_tokens(inputs, targets, -1, 0)
    print(enc_in)
    print(enc_targets)
    print()
    
    enc_in, enc_targets, rec_target = introduce_waiting_tokens_for_ae(inputs, targets, -1, 0)
    print(enc_in)
    print(enc_targets)
    print(rec_target)
    print()

#%%
    # inputs of length 6, targets of length 4
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6],
                                         [1, 2, 0, 0, 0, 0]], dtype=torch.long),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1],
                                              [1, 1, 0, 0, 0, 0]], dtype=torch.long)}
    targets = {'input_ids': torch.tensor([[11, 12, 13, 14],
                                          [11, 12, 13, 0, ]], dtype=torch.long),
               'attention_mask': torch.tensor([[1, 1, 1, 1],
                                               [1, 1, 1, 0]], dtype=torch.long)}
    # inputs, targets = introduce_waiting_tokens(inputs, inputs, -1, 0)
    enc_in, enc_targets, rec_target = introduce_waiting_tokens_for_ae(inputs, targets, -1, 0)
    print(enc_in)
    print(enc_targets)
    print(rec_target)

# %%
