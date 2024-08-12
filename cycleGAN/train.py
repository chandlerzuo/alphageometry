import os
import torch
import tqdm
import pandas as pd
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler, GPT2Config
from accelerate import Accelerator
from data_loader.csv_loader import NLFLDatasetFromCSV
from frozen_discriminator import PerplexityCalculator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from transformers.models.gpt2.modeling_gpt2 import GPT2Block

def main():
    num_epochs = 10
    validate_every = 100
    valid_for_batches = 10
    batch_size = 8
    valid_recon_save_path = '/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/'

    # Initialize Accelerator
    accelerator = Accelerator()
    model_name = 'gpt2'

    perplexity_calculator = PerplexityCalculator()

    # Load tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    config = GPT2Config()
    encoder = GPT2LMHeadModel(config=config)
    decoder = GPT2LMHeadModel(config=config)

    # Move models to GPU before wrapping with FSDP
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    encoder.to(device)
    decoder.to(device)

    # Wrap models with FSDP
    encoder = FSDP(encoder)
    decoder = FSDP(decoder)

    # Prepare dataset and dataloader
    dataset = NLFLDatasetFromCSV('/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv', split='train')
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = NLFLDatasetFromCSV('/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv', split='validation')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Move models to device
    encoder, decoder, perplexity_calculator = accelerator.prepare(encoder, decoder, perplexity_calculator)

    # Optimizers
    auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-5)

    # Learning rate scheduler
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=auto_enc_opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        pbar = tqdm.tqdm(train_dataloader)
        for batch_idx, batch in enumerate(pbar):
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding

            # Encode formal to natural
            enc_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            enc_inputs = {k: v.to(accelerator.device) for k, v in enc_inputs.items()}  # Ensure inputs are on the right device
            enc_outputs = encoder(**enc_inputs, output_hidden_states=True)

            # Decode natural to formal
            rec_outputs = decoder(inputs_embeds=enc_outputs.hidden_states[-1], labels=enc_inputs['input_ids'])
            recon_loss = rec_outputs.loss

            log_perplexity_loss = perplexity_calculator(enc_outputs.logits)

            # Total loss
            total_loss = recon_loss + 0 * log_perplexity_loss

            # Backpropagation
            accelerator.backward(total_loss)
            auto_enc_opt.step()
            lr_scheduler.step()
            auto_enc_opt.zero_grad()

            if batch_idx % validate_every == 0:
                # Validation
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    valid_iterator = iter(valid_dataloader)
                    val_recon_loss = 0
                    val_log_perplexity_loss = 0
                    for i, val_batch in enumerate(valid_iterator):
                        if i > valid_for_batches:
                            break
                        val_formal_texts = val_batch['formal']
                        val_natural_texts = val_batch['natural']

                        # Encode formal to natural
                        val_enc_inputs = tokenizer(val_formal_texts, return_tensors='pt', padding=True, truncation=True,
                                                   max_length=512)
                        val_enc_inputs = {k: v.to(accelerator.device) for k, v in val_enc_inputs.items()}
                        val_enc_outputs = encoder(**val_enc_inputs, output_hidden_states=True)

                        # Decode natural to formal
                        val_rec_outputs = decoder(inputs_embeds=val_enc_outputs.hidden_states[-1],
                                              labels=val_enc_inputs['input_ids'])
                        val_recon_loss += val_rec_outputs.loss

                        val_log_perplexity_loss += perplexity_calculator(val_enc_outputs.logits)

                        if i == 0:
                            # Convert logits to predicted token IDs
                            recon_token_ids = torch.argmax(val_rec_outputs.logits, dim=-1)
                            # import ipdb; ipdb.set_trace()
                            # Convert token IDs to text
                            recon_texts = tokenizer.batch_decode(recon_token_ids, skip_special_tokens=False)
                            df = pd.DataFrame({
                                'original': val_formal_texts,
                                'reconstructed': recon_texts
                            })
                            # Save the DataFrame to a CSV file
                            file_name = f'{epoch}_{batch_idx}_fl_fl.csv'
                            df.to_csv(os.path.join(valid_recon_save_path, file_name), index=False, encoding='utf-8')

                val_update = f'Average val_rec_l: {val_recon_loss / valid_for_batches:.3f}, ' \
                             f'Average val_log_perp_l: {val_log_perplexity_loss / valid_for_batches:.3f}'

            pbar.set_description(f'{val_update} log_perplexity_loss:{log_perplexity_loss:.3f}, recon_loss:{recon_loss:.3f}')
            encoder.train()
            decoder.train()

    print(f"Training completed. A total of {epoch} epoch(s)")


if __name__ == "__main__":
    main()
