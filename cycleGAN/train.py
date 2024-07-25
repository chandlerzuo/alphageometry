import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW, get_scheduler
from accelerate import Accelerator
from datasets import load_dataset
from frozen_discriminator import PerplexityCalculator


def main():
    num_epochs  = 10
    # Initialize Accelerator
    accelerator = Accelerator()

    perplexity_calculator = PerplexityCalculator('gpt2')

    # Load tokenizer and models
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    encoder = GPT2LMHeadModel.from_pretrained('gpt2')
    decoder = GPT2LMHeadModel.from_pretrained('gpt2')

    # Prepare dataset and dataloader
    dataset = load_dataset('your_dataset', split='train')  # Adjust accordingly
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Move models to device
    encoder, decoder, perplexity_calculator = accelerator.prepare(encoder, decoder, perplexity_calculator)

    # Optimizers
    auto_enc_opt = AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5)


    # Learning rate scheduler
    num_training_steps = num_epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=encoder_optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # Training loop
    encoder.train()
    decoder.train()
    for epoch in range(num_epochs):
        for batch in dataloader:
            formal_texts = batch['formal']
            natural_texts = batch['natural']

            # Encode formal to natural
            inputs = tokenizer(formal_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = encoder(**inputs, labels=inputs['input_ids'])
            encoder_loss = outputs.loss
            natural_generated_ids = outputs.logits.argmax(-1)
            natural_generated_texts = tokenizer.decode(natural_generated_ids, skip_special_tokens=True)

            # Decode natural to formal
            reconstructed_inputs = tokenizer(natural_generated_texts, return_tensors='pt', padding=True,
                                             truncation=True, max_length=512)
            reconstructed_outputs = decoder(**reconstructed_inputs, labels=inputs['input_ids'])
            decoder_loss = reconstructed_outputs.loss

            # Calculate perplexity for natural text generation
            perplexity = torch.exp(encoder_loss)

            # Total loss
            total_loss = encoder_loss + decoder_loss + perplexity

            # Backpropagation
            accelerator.backward(total_loss)
            auto_enc_opt.step()
            lr_scheduler.step()
            auto_enc_opt.zero_grad()

    print("Training completed.")


if __name__ == "__main__":
    main()
