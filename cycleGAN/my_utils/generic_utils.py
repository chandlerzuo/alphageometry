import torch
import os
import tqdm


class ProgressBar:
    def __init__(self, iterable_obj, accelerator):
        self.accelerator = accelerator
        self.local_rank = accelerator.state.process_index
        if self.local_rank == 0:
            self.pbar = tqdm.tqdm(iterable_obj)
        else:
            self.pbar = iterable_obj

    def __iter__(self):
        return self.pbar.__iter__()

    def __next__(self):
        return self.pbar.__next__()

    def update(self, n=1):
        if self.local_rank == 0:
            self.pbar.update(n)

    def set_description(self, desc):
        if self.local_rank == 0:
            self.pbar.set_description(desc)


def get_process_cuda_memory_info():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))  # Defaults to 0 if LOCAL_RANK is not set
    memory_info = {}

    if torch.cuda.is_available():
        device_info = {
            'device_name': torch.cuda.get_device_name(local_rank),
            'memory_allocated_gb': torch.cuda.memory_allocated(local_rank) / 1e9,  # Convert bytes to GB
            'memory_reserved_gb': torch.cuda.memory_reserved(local_rank) / 1e9,  # Convert bytes to GB
            'total_memory_gb': torch.cuda.get_device_properties(local_rank).total_memory / 1e9,  # Convert bytes to GB
        }

        # Calculate used and free memory
        free_memory = torch.cuda.memory_reserved(local_rank) / 1e9
        total_memory = torch.cuda.get_device_properties(local_rank).total_memory / 1e9
        used_memory = total_memory - free_memory

        device_info['used_memory_gb'] = used_memory
        device_info['free_memory_gb'] = free_memory

        # Peak memory usage
        mem_stats = torch.cuda.memory_stats(local_rank)
        device_info['peak_memory_allocated_gb'] = mem_stats['allocated_bytes.all.peak'] / 1e9
        device_info['peak_memory_reserved_gb'] = mem_stats['reserved_bytes.all.peak'] / 1e9

        memory_info[f"cuda:{local_rank}"] = device_info
    else:
        memory_info['error'] = "No CUDA devices available."

    return memory_info


def prit_proc0(msg):
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if local_rank == 0:
        print(msg)


def print_model_device_distribution(accelerator, model, model_name):
    device_params_count = {}  # Dictionary to hold device counts

    # Iterate through all parameters and count them per device
    for parameter in model.parameters():
        device = parameter.device
        if device in device_params_count:
            device_params_count[device] += parameter.numel()  # Add number of elements in parameter
        else:
            device_params_count[device] = parameter.numel()

    # Print the process rank and distribution of parameters across devices
    if accelerator.distributed_type.lower() == 'deepspeed':
        print('WARNING: Parameter distribution will not be visible like this!')
    print(f"Rank: {accelerator.state.process_index}, {model_name} is distributed across the following devices:")
    for device, count in device_params_count.items():
        print(f"Device: {device}, Number of Parameters: {count:,}")


if __name__ == '__main__':
    # Example usage
    gpu_memory_info = get_process_cuda_memory_info()
    print(gpu_memory_info)

