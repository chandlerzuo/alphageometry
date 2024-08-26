import torch
import os
import tqdm
import re


class ProgressBar:
    """Progress bar on main process only, otherwise normal iterable"""
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


def compress_text_forwaiting_and_eot_tokens(input_text, wait_token='<w>', eot_token='<|endoftext|>'):
    """
    Compresses texts by counting leading <w> tokens and trailing <eot> tokens,
    and then formats the string accordingly.

    Example:
        input: '<w> <w> A B C = Triangle A B C <|endoftext|><|endoftext|><|endoftext|>'
        output: '2<w> A B C = Triangle A B C 3<|endoftext|>'
    """
    # Adjusted regex to properly match the required pattern
    # Using non-capturing groups for tokens and capturing groups for counts and main text
    pattern = r'^(' + re.escape(wait_token) + r'\s*)+'
    match = re.match(pattern, input_text)
    if match:
        # Count occurrences of each token type
        wait_section = match.group() or ''  # Captured leading wait tokens section
        count_w = wait_section.count(wait_token)  # Count occurrences of wait_token
        rest_of_text = input_text[match.span()[1]:]  # Extract the main text content
        count_eot = (len(rest_of_text) - len(rest_of_text.rstrip(eot_token))) // len(eot_token)
        # Format output string with counts
        result_text = f"{count_w}{wait_token} {rest_of_text.rstrip(eot_token)} {count_eot}{eot_token}"
        return result_text
    return input_text  # Return original text if no matches


def batch_compress_text_forwaiting_and_eot_tokens(input_list_text, wait_token='<w>', eot_token='<|endoftext|>'):
    return [compress_text_forwaiting_and_eot_tokens(text.replace('\n', '<nl>'),
                                                    wait_token, eot_token) for text in input_list_text]


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


def print_proc0(msg):
    """print only on process 0"""
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
    # gpu_memory_info = get_process_cuda_memory_info()
    # print(gpu_memory_info)

    print(compress_text_forwaiting_and_eot_tokens(
        '<w> <w><w> A B C = Triangle A B C <|endoftext|><|endoftext|><|endoftext|>'))
    print(compress_text_forwaiting_and_eot_tokens('<w><w> A B C = Triangle A B C </s></s>', eot_token='</s>'))

