# Launch command
```
accelerate launch --config_file=configs/cycleGAN.yaml --machine_rank=1 train.py
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