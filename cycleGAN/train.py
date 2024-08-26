import os
import math
from torch.utils.data import DataLoader

from accelerate import Accelerator
from data_loader.csv_loader import NLFLDatasetFromCSV
from data_loader.json_loader import NLFLDatasetFromJSONL
from model_preperation import load_model
from transformers import AdamW, get_scheduler
from torch.utils.data.distributed import DistributedSampler
from my_utils.generic_utils import get_process_cuda_memory_info, prit_proc0, print_model_device_distribution, \
    ProgressBar
from my_utils.training_utils import compute_validation, introduce_waiting_tokens_for_ae, introduce_waiting_tokens, \
    Checkpointer, prepare_inputs


def main(args):
    wait_token = '<w>'
    mdl_dir = args.model_name
    if args.decoder_only:
        mdl_dir += '_dec_only'
    elif args.encoder_only:
        mdl_dir += '_enc_only'
    else:
        mdl_dir += '_full_ae'
    if args.use_perplexity_loss:
        mdl_dir += '+perplexity'
    mdl_output_home = os.path.join(args.output_path, mdl_dir)
    valid_recon_save_path = os.path.join(mdl_output_home, 'validation_outputs')
    chkpt_dir = os.path.join(mdl_output_home, 'checkpoints')

    checkpointer = Checkpointer(chkpt_dir)

    # Initialize Accelerator
    accelerator = Accelerator()
    if accelerator.is_main_process:
        os.makedirs(valid_recon_save_path, exist_ok=True)
        os.makedirs(chkpt_dir, exist_ok=True)

    # this micro batching might not be optimal!
    if accelerator.distributed_type.lower() == 'deepspeed':
        # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] \
        #     = args.batch_size // accelerator.num_processes
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] \
            = 1

    # Load tokenizer and models
    ae_model, tokenizer, wait_id = load_model(args.model_name, wait_token=wait_token, use_pretrained=args.is_pretrained,
                                              use_perplexity_loss=args.use_perplexity_loss,
                                              decoder_only=args.decoder_only, encoder_only=args.encoder_only)

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
    v_rephrased_dat = \
        NLFLDatasetFromJSONL(
            '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/rephrased-nl_fl_dataset_all.jsonl',
            split='validation', overfitting=args.overfitting)

    # Validation dataset (can use a different or similar sampler depending on the setup)
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=accelerator.num_processes,
                                       rank=accelerator.process_index)
    v_rephrased_samp = DistributedSampler(v_rephrased_dat, num_replicas=accelerator.num_processes,
                                       rank=accelerator.process_index)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)
    v_rephrased_ldr = DataLoader(v_rephrased_dat, batch_size=args.batch_size, sampler=v_rephrased_samp)

    # Optimizers
    # auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=2e-5)
    auto_enc_opt = AdamW(ae_model.parameters(), lr=2e-5 * 4)

    # Learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        # name="linear",
        name="cosine",
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
    enc_loss = recon_loss = 0
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

            # perform padding according to training mode
            ae_inputs, encoder_target, ae_target = prepare_inputs(
                encoder_only=args.encoder_only, decoder_only=args.decoder_only, formal_lang=enc_inputs,
                natural_lang=enc_label, wait_id=wait_id, pad_token_id=tokenizer.pad_token_id)

            # import ipdb;ipdb.set_trace()
            # encode and decode
            enc_outputs, rec_outputs, log_perplexity_loss, _ = \
                ae_model(**ae_inputs, output_hidden_states=True, encoder_target=encoder_target,
                         recon_target=ae_target)

            # Total loss. This unfortunately can't be initialized to 0 and all others added! It complains about double
            # backwardpass
            total_loss = 0
            if rec_outputs is not None:
                recon_loss = rec_outputs.loss
                total_loss = recon_loss
            if enc_outputs is not None:
                total_loss += args.enc_loss_weight * enc_outputs.loss
                enc_loss = enc_outputs.loss.item()

            total_loss += log_perplexity_loss
            # Backpropagation
            accelerator.backward(total_loss)
            # import ipdb; ipdb.set_trace()
            auto_enc_opt.step()
            lr_scheduler.step()
            auto_enc_opt.zero_grad()

            if batch_idx % args.validate_every == args.validate_every - 1:
                val_update = compute_validation(accelerator, ae_model, args, epoch * len(train_dataloader) + batch_idx,
                                                epoch, tokenizer, valid_dataloader, v_rephrased_ldr,
                                                valid_recon_save_path, wait_id, args.chkpt_bst_mdl_every, checkpointer)

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
    parser.add_argument('--chkpt_bst_mdl_every', type=int, default=10,
                        help='Checkpoint model every these many validation steps if validation result improved. '
                             'Negative value skips this')
    parser.add_argument('--output_path', type=str,
                        default='/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/',
                        help='path to save training stats and models')
    parser.add_argument('--grounding_prob', type=float, default=0.5, help='probability of encoder grounding!')
    parser.add_argument('--enc_loss_weight', type=float, default=2, help='probability of encoder grounding!')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name to load, e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',"
                                                       "'meta-llama/Meta-Llama-3.1-8B', "
                                                       "'meta-llama/Llama-2-7b-hf'")
    parser.add_argument('--overfitting', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--is_pretrained', type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)
    parser.add_argument('--decoder_only', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--encoder_only', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--use_perplexity_loss',
                        type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)

    args = parser.parse_args()
    args.chkpt_bst_mdl_every *= args.validate_every
    if args.decoder_only:
        assert not args.use_perplexity_loss
        assert args.grounding_prob >= 1  # you need the grounding always as these are the inputs
    if args.encoder_only:
        assert args.grounding_prob >= 1, f'We need the natural language targets when using encoder only model.'
    prit_proc0(args)
    main(args)
