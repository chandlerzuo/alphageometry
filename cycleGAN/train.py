import math
from torch.utils.data import DataLoader

from accelerate import Accelerator
from data_loader.csv_loader import NLFLDatasetFromCSV
from model_preperation import load_model
from transformers import AdamW, get_scheduler
from torch.utils.data.distributed import DistributedSampler
from my_utils.generic_utils import get_process_cuda_memory_info, prit_proc0, print_model_device_distribution, \
    ProgressBar
from my_utils.training_utils import compute_validation, introduce_waiting_tokens_for_ae


def main(args):
    wait_token = '<w>'
    valid_recon_save_path = '/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/'

    # Initialize Accelerator
    accelerator = Accelerator()
    # this microbatching might not be optimal!
    if accelerator.distributed_type.lower() == 'deepspeed':
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] \
            = args.batch_size // accelerator.num_processes


    # Load tokenizer and models
    ae_model, tokenizer, wait_id = load_model(args.model_name, wait_token=wait_token, use_pretrained=args.is_pretrained)

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
    # auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-5)
    auto_enc_opt = AdamW(ae_model.parameters(), lr=2e-5)

    # Learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=auto_enc_opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # import ipdb; ipdb.set_trace()
    ae_model, auto_enc_opt, tokenizer, lr_scheduler = accelerator.prepare(ae_model, auto_enc_opt, tokenizer,
                                                                          lr_scheduler)
    # encoder, tokenizer, auto_enc_opt, lr_scheduler, train_dataloader = \
    #     accelerator.prepare([encoder, tokenizer, auto_enc_opt, lr_scheduler, train_dataloader])

    # Training loop
    ae_model.train()
    val_update = f'val_update: None'
    for epoch in range(args.num_epochs):
        pbar = ProgressBar(train_dataloader, accelerator)
        for batch_idx, batch in enumerate(pbar):
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding
            # Encode formal to natural
            enc_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            enc_inputs = {k: v.to(accelerator.device) for k, v in
                          enc_inputs.items()}  # Ensure inputs are on the right device

            # This has to be deterministic else some process will wait for the others!
            if batch_idx % math.ceil(1/(args.grounding_prob + 1e-7)) == 0:
                enc_label = tokenizer(natural_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
                enc_label = {k: v.to(accelerator.device) for k, v in
                              enc_label.items()}  # Ensure inputs are on the right device
            else:
                enc_label = None

            enc_inputs, encoder_target, recon_target = introduce_waiting_tokens_for_ae(
                enc_inputs, enc_label, wait_id, padding_id=tokenizer.pad_token_id)

            # import ipdb;ipdb.set_trace()
            # encode and decode
            enc_outputs, rec_outputs, log_perplexity_loss, _ = \
                ae_model(**enc_inputs, output_hidden_states=True, encoder_target=encoder_target,
                         recon_target=recon_target)

            # print_model_device_distribution(accelerator, ae_model, 'ae_model')
            # exit(-1)
            recon_loss = rec_outputs.loss

            # Total loss
            total_loss = recon_loss + 0 * log_perplexity_loss
            if enc_outputs.loss is not None:
                total_loss += enc_outputs.loss
                enc_loss = enc_outputs.loss.item()
            else:
                enc_loss = 0
            # Backpropagation
            accelerator.backward(total_loss)
            auto_enc_opt.step()
            lr_scheduler.step()
            auto_enc_opt.zero_grad()

            if batch_idx % args.validate_every == 0:
                val_update = compute_validation(accelerator, ae_model, args, batch_idx, epoch,
                                                tokenizer, valid_dataloader, valid_recon_save_path, wait_id)

            pbar.set_description(f'{val_update} log_perp_loss:{log_perplexity_loss:.3f}, '
                                 f'recon_loss:{recon_loss:.3f}, enc_loss:{enc_loss:.3f}')

    print(f"Training completed. A total of {epoch} epoch(s)")


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=100)
    parser.add_argument('--valid_for_batches', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32, help='Batch per GPU!')
    parser.add_argument('--grounding_prob', type=float, default=0.5, help='probability of encoder grounding!')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name to load, e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',"
                                                       "'meta-llama/Meta-Llama-3.1-8B', "
                                                       "'meta-llama/Llama-2-7b-hf'")
    parser.add_argument('--overfitting', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--is_pretrained', type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)

    args = parser.parse_args()
    prit_proc0(args)
    main(args)
