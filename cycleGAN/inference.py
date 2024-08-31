import os
import argparse
from accelerate import Accelerator
import torch
from transformers import AutoTokenizer
import deepspeed
import json
from model_preparation import load_model


def load_model_for_inference(checkpoint_path, wait_token='<w>'):
    train_cmdlargs_path = os.path.join(checkpoint_path, 'cmd_args.json')
    with open(train_cmdlargs_path, 'r') as file:
        data = json.load(file)

    ae_model, tokenizer, wait_id = load_model(data.model_name, wait_token=wait_token, use_pretrained=False,
                                              use_perplexity_loss=False, use_decoder=data.use_decoder,
                                              use_encoder=data.use_encoder)
    ae_model.from_pretrained(checkpoint_path)
    return ae_model


def generate_text(model, tokenizer, input_text, max_length, num_beams, do_sample, top_k, top_p):
    # Configure generation parameters
    generation_args = {
        'max_length': max_length,
        'num_beams': num_beams,
        'do_sample': do_sample,
        'top_k': top_k,
        'top_p': top_p,
        'early_stopping': True if num_beams > 1 else False
    }

    # Encode input text
    inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True).to(model.device)

    # Generate output using specified strategy
    with torch.no_grad():
        output = model.generate(**inputs, **generation_args)

    # Decode and return output text
    return tokenizer.decode(output[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Generate text from a pretrained model")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the DeepSpeed model checkpoint directory")
    parser.add_argument("--input_text", type=str, required=True, help="Input text to generate text from")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length of the generated text")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search")
    parser.add_argument("--do_sample", action='store_true', help="Enable sampling for generation")
    parser.add_argument("--top_k", type=int, default=0, help="Top-K sampling")
    parser.add_argument("--top_p", type=float, default=0.0, help="Top-P (nucleus) sampling")

    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("allenai/llama")

    # Load model
    model = load_model_for_inference(checkpoint_path=args.checkpoint_path)

    # Generate text
    generated_text = generate_text(model, tokenizer, args.input_text, args.max_length, args.num_beams, args.do_sample,
                                   args.top_k, args.top_p)
    print(generated_text)


if __name__ == "__main__":
    main()
