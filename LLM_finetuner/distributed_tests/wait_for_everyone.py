# Example script to run with accelerate

from accelerate import Accelerator
import torch


accelerator = Accelerator()
print(accelerator.state)
print("Waiting for everyone: ")
accelerator.wait_for_everyone()
print("Everyone is ready")

process_tensor = torch.tensor([accelerator.process_index]).cuda() # cuda required for gather()
gathered_tensor = accelerator.gather(process_tensor)
print(f"Gathered tensor: {gathered_tensor}")
# todo: gather