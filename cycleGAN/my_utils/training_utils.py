import torch
import pandas as pd
import os
import torch.nn.functional as F
try:
    from .generic_utils import batch_compress_text_forwaiting_and_eot_tokens
except ImportError:
    from generic_utils import batch_compress_text_forwaiting_and_eot_tokens


def compute_validation(accelerator, ae_model, args, batch_idx, epoch, tokenizer,
                       valid_dataloader, valid_recon_save_path, wait_id):
    # Validation
    ae_model.eval()
    with torch.no_grad():
        valid_iterator = iter(valid_dataloader)
        val_recon_loss = 0
        val_enc_loss = 0
        val_log_perplexity_loss = 0
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

            val_enc_inputs, val_encoder_target, val_recon_target = introduce_waiting_tokens_for_ae(
                val_enc_inputs, enc_label, wait_id, padding_id=tokenizer.pad_token_id)

            dec_natural, _ = \
                introduce_waiting_tokens(enc_label, enc_label, wait_id, padding_id=tokenizer.pad_token_id)

            val_enc_outputs, val_rec_outputs, val_log_perplexity_loss_batch, decoded_from_natural = \
                ae_model(**val_enc_inputs, output_hidden_states=True, recon_target=val_recon_target,
                         encoder_target=val_encoder_target, decode_natural=dec_natural)

            decoded_token_ids = torch.argmax(decoded_from_natural.logits, dim=-1)
            decoded_frm_nl_texts = tokenizer.batch_decode(decoded_token_ids, skip_special_tokens=False)

            val_recon_loss += val_rec_outputs.loss
            val_enc_loss += val_enc_outputs.loss
            val_log_perplexity_loss += val_log_perplexity_loss_batch

            if i == 0:
                # print(get_process_cuda_memory_info())
                # Convert logits to predicted token IDs
                recon_token_ids = torch.argmax(val_rec_outputs.logits, dim=-1)
                # Convert token IDs to text
                recon_texts = tokenizer.batch_decode(recon_token_ids, skip_special_tokens=False)
                # Get encoded text
                enc_tok_ids = torch.argmax(val_enc_outputs.logits, dim=-1)
                enc_texts = tokenizer.batch_decode(enc_tok_ids, skip_special_tokens=False)
                val_formal_texts_2_save = tokenizer.batch_decode(val_recon_target, skip_special_tokens=False)
                val_natural_target = tokenizer.batch_decode(val_encoder_target, skip_special_tokens=False)
                df = pd.DataFrame({
                    'formal_target': batch_compress_text_forwaiting_and_eot_tokens(val_formal_texts_2_save,
                                                                                   eot_token=tokenizer.eos_token),
                    'formal_reconstructed': batch_compress_text_forwaiting_and_eot_tokens(
                        recon_texts, eot_token=tokenizer.eos_token),
                    'natural_created': batch_compress_text_forwaiting_and_eot_tokens(enc_texts,
                                                                                     eot_token=tokenizer.eos_token),
                    'natural_target': batch_compress_text_forwaiting_and_eot_tokens(val_natural_target,
                                                                                    eot_token=tokenizer.eos_token),
                    'formal_generated': batch_compress_text_forwaiting_and_eot_tokens(decoded_frm_nl_texts,
                                                                                      eot_token=tokenizer.eos_token)
                })
                # Save the DataFrame to a CSV file
                file_name = f'{epoch}_{batch_idx}_fl_fl.csv'
                df.to_csv(os.path.join(valid_recon_save_path, file_name), index=False, encoding='utf-8')
    val_update = f'a_v_rec_l: {val_recon_loss / args.valid_for_batches:.3f}, ' \
                 f'a_v_enc_l: {val_enc_loss / args.valid_for_batches:.3f}, ' \
                 f'a_v_l_perp_l: {val_log_perplexity_loss / args.valid_for_batches:.3f}'
    ae_model.train()
    return val_update


def introduce_waiting_tokens(inputs, targets, wait_token_id, padding_id):
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


if __name__ == '__main__':
    inputs = {'input_ids': torch.tensor([[1, 2, 3, 4, 5, 6],
                                         [1, 2, 0, 0, 0, 0]], dtype=torch.long),
              'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1],
                                              [1, 1, 0, 0, 0, 0]], dtype=torch.long)}
    targets = {'input_ids': torch.tensor([[11, 12, 13, 14, 15, 16, 17],
                                          [11, 12, 13, 0, 0, 0, 0]], dtype=torch.long),
               'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1],
                                               [1, 1, 1, 0, 0, 0, 0]], dtype=torch.long)}
    # inputs, targets = introduce_waiting_tokens(inputs, inputs, -1, 0)
    enc_in, enc_targets, rec_target = introduce_waiting_tokens_for_ae(inputs, targets, -1, 0)
    print(enc_in)
    print(enc_targets)
    print(rec_target)

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
