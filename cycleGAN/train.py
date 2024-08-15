import os
import torch
import tqdm
import pandas as pd
from torch.utils.data import DataLoader

from accelerate import Accelerator
from data_loader.csv_loader import NLFLDatasetFromCSV
from frozen_discriminator import PerplexityCalculator
from model_preperation import load_model
from transformers import AdamW, get_scheduler
from torch.utils.data.distributed import DistributedSampler
from my_utils import get_process_cuda_memory_info, prit_proc0, print_model_device_distribution

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy



def main(args):
    valid_recon_save_path = '/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/'

    # Initialize Accelerator
    accelerator = Accelerator()

    perplexity_calculator = PerplexityCalculator()

    # Load tokenizer and models
    tokenizer, encoder, decoder = load_model(args.model_name, use_pretrained=args.is_pretrained)

    # Move models to device
    encoder, decoder, tokenizer, perplexity_calculator = \
        accelerator.prepare(encoder, decoder, tokenizer, perplexity_calculator)

    # Prepare dataset and dataloader
    dataset = \
        NLFLDatasetFromCSV('/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv', split='train',
                           overfitting=args.overfitting)
    # Create a DistributedSampler for the dataset
    train_sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes, rank=accelerator.process_index)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)

    valid_dataset = \
        NLFLDatasetFromCSV('/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv', split='validation',
                           overfitting=args.overfitting)

    # Validation dataset (can use a different or similar sampler depending on the setup)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=accelerator.num_processes,
                                       rank=accelerator.process_index)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)

    # Optimizers
    auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-5)

    # Learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=auto_enc_opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    encoder.train()
    decoder.train()
    for epoch in range(args.num_epochs):
        pbar = tqdm.tqdm(train_dataloader)
        for batch_idx, batch in enumerate(pbar):
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding

            # Encode formal to natural
            enc_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)

            enc_outputs = encoder(**enc_inputs, output_hidden_states=True)

            print_model_device_distribution(accelerator, encoder, 'encoder')
            # exit(-1)

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

            if batch_idx % args.validate_every == 0:
                # Validation
                encoder.eval()
                decoder.eval()
                with torch.no_grad():
                    valid_iterator = iter(valid_dataloader)
                    val_recon_loss = 0
                    val_log_perplexity_loss = 0
                    for i, val_batch in enumerate(valid_iterator):
                        if i > args.valid_for_batches:
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
                            # print(get_process_cuda_memory_info())
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

                val_update = f'Average val_rec_l: {val_recon_loss / args.valid_for_batches:.3f}, ' \
                             f'Average val_log_perp_l: {val_log_perplexity_loss / args.valid_for_batches:.3f}'

            pbar.set_description(f'{val_update} log_perplexity_loss:{log_perplexity_loss:.3f}, recon_loss:{recon_loss:.3f}')
            encoder.train()
            decoder.train()

    print(f"Training completed. A total of {epoch} epoch(s)")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--valid_for_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name to load, e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',"
                                                       "'meta-llama/Meta-Llama-3.1-8B', "
                                                       "'meta-llama/Llama-2-7b-hf'")
    parser.add_argument('--use_FSDP', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--overfitting', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--is_pretrained', type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)

    args = parser.parse_args()
    prit_proc0(args)
    main(args)
