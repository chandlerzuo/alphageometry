## Caveat
If you are getting tensor size missmatch error, while sending in `enc_resume_path` and/or `dec_resume_path`, then you 
have both `model.safetensors` and `model-00001-of-00004.safetensors` inside the checkpoint directory. Simply delete the
`model.safetensors`. Check though the model is indeed saved in several parts.
# Launch command

```
accelerate launch --config_file=accelerate_configs/multi_machine_config.yml --main_process_ip=0.0.0.0 \
--num_processes=1 train.py --use_encoder=False --use_decoder=True --batch_size=1 --grounding_prob=1 \
--model_name=gpt2

```
### Restore encoder and decoder and run AE training
```
accelerate launch --config_file=accelerate_configs/multi_machine_config.yml --main_process_ip=0.0.0.0 \
--num_processes=1 train.py --use_encoder=True --use_decoder=True --batch_size=1 --grounding_prob=1 \
--model_name=gpt2 --validate_every=1 --chkpt_bst_mdl_every=1 --use_perplexity_loss=False \
--enc_resume_path=<root>/gpt2_enc_only/checkpoints/ \
--dec_resume_path=<root>/gpt2_dec_only/checkpoints/
```

## Inferance
```commandline
python inference.py --ckpt <ckpt_root>/meta-llama/Meta-Llama-3.1-8B_dec_only_4/checkpoints/
```
### Padding
Two modes 
1. copy_target: Copy the target in the input
2. pad_tok: use noninformative padding token

formal -> encoder -> natural -> decoder -> formal_recon

example input and target for the Encode only setup

### Example input and target for the Encode only setup

**Mode copy_target**

`f f f t t t t` : Encoder input

`w w w t t t t` : Encoder target

**Mode pad_tok**

`f f f p p p p` : Encoder input

`w w w t t t t` : Encoder target

### Example input and target for the Decode only setup

**Mode copy_target**

`n n n t t t t` : Decoder input

`w w w t t t t` : Decoder target

**Mode pad_tok**

`n n n p p p p` : Decoder input

`w w w t t t t` : Decoder target



### Example input and target for the AE setup

`f f f t t t t p p p` : Encoder input

`w w w t t t t p p p` : Encoder target = Decoder input

`w w w w w w w f f f` : Decoder target