import os
import math
from pathlib import Path
import sys
from torch.utils.data import DataLoader

from accelerate import Accelerator
from hf_dataset import AlwaysSameElementDataset, CombinedDataset, MixedDatasetSampler, prepare_data
from model_preparation import load_model
from transformers import get_scheduler
from torch.optim import AdamW
from my_utils.generic_utils import get_process_cuda_memory_info, print_proc0, print_model_device_distribution, \
    ProgressBar, get_project_out_dir

from my_utils.training_utils import as_dict, create_val_metrics_string, Checkpointer, prepare_formal_natural_inputs,\
    run_validation

import accelerate

import wandb
os.environ.setdefault("WANDB_PROJECT", "alphageom_new")
wandb.require("core")

from utils import get_comma_separated_strings, get_hostname, get_username
from my_utils.hf_wrapper import debug_on_error

@debug_on_error
def main(args):
    wait_token = '<w>'

    # Initialize Accelerator
    accelerator = Accelerator(log_with="all")
    if get_username() == "mmordig":
        accelerator.init_trackers("alphageom_autoencoder", config=vars(args))
        # pass
    valid_recon_save_path, chkpt_dir = get_project_out_dir(args, accelerator.is_main_process)
    print(f"Outputting to dirs {valid_recon_save_path} and {chkpt_dir}")
    checkpointer = Checkpointer(chkpt_dir, args)

    seed = 42
    accelerate.utils.set_seed(seed)

    # Load tokenizer and models
    ae_model, tokenizer, wait_id = load_model(args.model_name, wait_token=wait_token, use_pretrained=args.is_pretrained,
                                              use_perplexity_loss=args.use_perplexity_loss,
                                              use_decoder=args.use_decoder, use_encoder=args.use_encoder)

    wrap_dataset = AlwaysSameElementDataset if args.overfitting else lambda x: x
    # Prepare dataset and dataloader
    non_rephrased_dataset = prepare_data(
        args.dataset_dir / 'nl_fl.csv', seed=seed, nrows=args.nrows_nonrephrased
    )
    rephrased_dataset = prepare_data(
        args.dataset_dir / 'rephrased-nl_fl_dataset_all.jsonl', seed=seed, nrows=args.nrows_rephrased, 
        colnames={"formal": "fl_statement", "natural": "rephrase"}
    )
    
    if args.rephrased_ratio > 0:
        print(f"Using {args.rephrased_ratio} rephrased data")
        train_datasets = [non_rephrased_dataset["train"], rephrased_dataset["train"]]
        assert set(train_datasets[0].column_names) == set(train_datasets[1].column_names)
        train_dataset = CombinedDataset(*train_datasets)
        train_sampler = MixedDatasetSampler(
            [len(x) for x in train_datasets], args.rephrased_ratio
        )
        del train_datasets
    else:
        train_dataset = non_rephrased_dataset["train"]
        train_sampler = None
    train_dataset = wrap_dataset(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True,
                                  shuffle=False)

    val_dataset = wrap_dataset(non_rephrased_dataset["validation"])
    val_rephrased_dat = wrap_dataset(rephrased_dataset["validation"])

    valid_sampler = None # not needed, handled by accelerate
    val_rephrased_samp = None
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=valid_sampler, pin_memory=True,
                                shuffle=False)
    val_rephrased_dataloader = DataLoader(val_rephrased_dat, batch_size=args.batch_size, sampler=val_rephrased_samp)

    # Optimizers
    optimizer = AdamW(ae_model.parameters(), lr=2e-5 * 4)
    
    # Learning rate scheduler
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        # name="linear",
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # import ipdb; ipdb.set_trace()
    ae_model, optimizer, lr_scheduler, train_dataloader, val_dataloader, val_rephrased_dataloader = accelerator.prepare(
        ae_model, optimizer, lr_scheduler, train_dataloader, val_dataloader, val_rephrased_dataloader
    )

    # Training loop
    ae_model.train()
    val_update_str = f'val_update: N/A'
    for epoch in range(args.num_epochs):
        pbar = ProgressBar(train_dataloader, accelerator)
        for batch_idx, batch in enumerate(pbar):
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding
            
            formal_inputs, natural_inputs = prepare_formal_natural_inputs(
                formal_texts, natural_texts, tokenizer=tokenizer,
                return_natural_inputs=batch_idx % math.ceil(1/(args.grounding_prob + 1e-7)) == 0
            )

            ae_model.train()
            model_outputs, _ = ae_model(formal_inputs=formal_inputs, natural_inputs=natural_inputs,
                                        wait_token_id=wait_id, pad_token_id=tokenizer.pad_token_id)
            total_loss = model_outputs.total_loss(enc_loss_weight=args.enc_loss_weight)

            accelerator.backward(total_loss)
            # import ipdb; ipdb.set_trace()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            train_metrics = {
                "log_perp_loss": model_outputs.log_perplexity_loss or 0, 
                "recon_loss": model_outputs.decoder_loss(), 
                "enc_loss": model_outputs.encoder_loss(),
            }

            if batch_idx % args.validate_every == args.validate_every - 1:
                
                df_filename = os.path.join(valid_recon_save_path, f'{epoch}_{batch_idx}_fl_fl.csv')
                val_metrics = run_validation(
                    model=ae_model, accelerator=accelerator, tokenizer=tokenizer, 
                    non_rephrased_dataloader=val_dataloader, rephrased_dataloader=val_rephrased_dataloader,
                    df_filename=df_filename,
                    model_kwargs=as_dict(wait_token_id=wait_id, pad_token_id=tokenizer.pad_token_id),
                    max_num_batches=args.valid_for_batches,
                )
                val_update_str = "val_update: " + create_val_metrics_string(val_metrics)
                accelerator.log(
                    {f"val/{k}": v for k, v in val_metrics.items()}
                    |
                    {f"train/{k}": v for k, v in train_metrics.items()}
                    |
                    {f"epoch": epoch + batch_idx / len(train_dataloader),
                     "step": epoch * len(train_dataloader) + batch_idx}
                )
                
                ae_model.train()
                if (batch_idx + 1) % args.chkpt_bst_mdl_every == 0:
                    loss_for_saving = val_metrics["enc_loss"] + val_metrics["recon_loss"]
                    checkpointer.checkpoint(accelerator, ae_model, loss_for_saving)
                
            train_update_str = " ".join([f"{k}:{v:.3f}" for k, v in train_metrics.items()])
            pbar.set_description(f'Epoch {epoch+1}/{args.num_epochs}: {val_update_str} {train_update_str}')

    accelerator.end_training()
    print(f"Training completed. A total of {epoch+1} epoch(s)")


if __name__ == "__main__":
    if get_hostname() == "mikado":
        os.environ["ALPHA_GEOM_DATASET"] = "/home/mmordig/reinforcement/alphageometry/cycleGAN/runs/datasets/arithmetic"

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--validate_every', type=int, default=100, help='Validate every these many training steps '
                                                                        '(reset per epoch)')
    parser.add_argument('--valid_for_batches', type=int, default=10, help='Validate for these many batches')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU! when deepspeed is enabled '
                                                                   '(for model pipelining), it is divided by the number'
                                                                   ' of gpus for microbatching')
    parser.add_argument('--dataset_dir', type=Path, default=None, help='Path to dataset directory')
    parser.add_argument('--chkpt_bst_mdl_every', type=int, default=10,
                        help='Checkpoint model every these many validation (!) steps if validation result improved. '
                             'Negative value skips this')
    parser.add_argument('--output_path', type=str,
                        default=None,
                        help='path to save training stats and models')
    parser.add_argument('--grounding_prob', type=float, default=0.5, help='introduce encoder NL labels every ceil(1/x)'
                                                                          ' batches')
    parser.add_argument('--enc_loss_weight', type=float, default=2, help='scale encoder loss by this factor')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf',
                        help="Model name to load, e.g., 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl',"
                                                       "'meta-llama/Meta-Llama-3.1-8B', "
                                                       "'meta-llama/Llama-2-7b-hf'")
    parser.add_argument('--overfitting', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False,
                        help="whether to overfit on a single batch (same across all GPUs)")
    parser.add_argument('--is_pretrained', type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)
    parser.add_argument('--use_decoder', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--use_encoder', type=lambda x: True if x.lower() in ['true', '1'] else False, default=False)
    parser.add_argument('--use_perplexity_loss',
                        type=lambda x: True if x.lower() in ['true', '1'] else False, default=True)
    parser.add_argument('--nrows_nonrephrased', type=int, default=None,
                        help='Number of rows to load from the non-rephrased dataset before splitting into '
                             'train/val/test, defaults to all')
    parser.add_argument('--nrows_rephrased', type=int, default=None,
                        help='Number of rows to load from the rephrased dataset before splitting into '
                             'train/val/test, defaults to all')
    parser.add_argument('--rephrased_ratio', type=float, default=0, help='Ratio of picking from rephrased dataset')

    print(f"Unparsed arguments: {get_comma_separated_strings(sys.argv)}")
    args = parser.parse_args()
    
    if args.output_path is None:
        args.output_path = os.environ.get("ALPHA_GEOM_OUTPUT_PATH",
                                          '/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/')
        if get_username() == "mmordig":
            args.output_path = "runs/"
        print(f"Output path not provided, using {args.output_path}")

    if args.dataset_dir is None:
        args.dataset_dir = Path(os.environ.get("ALPHA_GEOM_DATASET",
                                               '/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/'))
    
    assert 0 <= args.rephrased_ratio, f"got invalid {args.rephrased_ratio=}"
    
    args.chkpt_bst_mdl_every *= args.validate_every
    if args.use_decoder and not args.use_encoder:
        assert not args.use_perplexity_loss
        assert args.grounding_prob >= 1  # you need the grounding always as these are the inputs

    if args.use_encoder and not args.use_decoder:
        assert args.grounding_prob >= 1, f'We need the natural language targets when using encoder only model.'

    print_proc0(f"Got arguments: {args}")
    main(args)
