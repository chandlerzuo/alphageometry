import tqdm
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from accelerate import Accelerator
from data_loader.csv_loader import NLFLDatasetFromCSV
from frozen_discriminator import PerplexityCalculator


def main():
    num_epochs  = 10
    # Initialize Accelerator
    accelerator = Accelerator()
    model_name = 'gpt2'

    perplexity_calculator = PerplexityCalculator()

    # Load tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    encoder = GPT2LMHeadModel.from_pretrained(model_name)
    decoder = GPT2LMHeadModel.from_pretrained(model_name)

    # Prepare dataset and dataloader
    dataset = NLFLDatasetFromCSV('/is/cluster/fast/scratch/pghosh/dataset/alpha_geo/geometry/nl_fl.csv',
                                 split='train')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Move models to device
    encoder, decoder, perplexity_calculator = accelerator.prepare(encoder, decoder, perplexity_calculator)

    # Optimizers
    auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5)

    # Learning rate scheduler
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=auto_enc_opt,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    encoder.train()
    decoder.train()
    pbar = tqdm.tqdm(range(num_epochs))
    for epoch in pbar:
        for batch in dataloader:
            formal_texts = batch['formal']
            natural_texts = batch['natural']  # perhaps sometimes we should use it for grounding

            # Encode formal to natural
            enc_inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            enc_outputs = encoder(**enc_inputs, output_hidden_states=True)

            # Decode natural to formal
            rec_outputs = decoder(inputs_embeds=enc_outputs.hidden_states[-1], labels=enc_inputs.input_ids)
            recon_loss = rec_outputs.loss

            log_perplexity_loss = perplexity_calculator(enc_outputs.logits)

            # Total loss
            total_loss = recon_loss + log_perplexity_loss

            # Backpropagation
            accelerator.backward(total_loss)
            auto_enc_opt.step()
            lr_scheduler.step()
            auto_enc_opt.zero_grad()
            pbar.set_description(f'log_perplexity_loss:{log_perplexity_loss:3f}, recon_loss:{recon_loss:3f}')

    print(f"Training completed. Atotal of {epoch} epoch(s)")


if __name__ == "__main__":
    main()
