import os
import shutil
import argparse


def cleanup_model_dirs(base_dir):
    """
    This function goes through each model directory inside the base directory and deletes any directories
    that do not contain non-empty 'checkpoints' and 'validation_outputs' directories.

    Args:
    - base_dir (str): The base directory containing model directories.
    """
    # Get all directories in the base directory
    model_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    # Iterate over each model directory
    for model_dir in model_dirs:
        # Define paths for 'checkpoints' and 'validation_outputs'
        checkpoint_dir = os.path.join(model_dir, 'checkpoints')
        validation_outputs_dir = os.path.join(model_dir, 'validation_outputs')

        # Check if both 'checkpoints' and 'validation_outputs' directories exist and are non-empty
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir) and \
                os.path.exists(validation_outputs_dir) and os.listdir(validation_outputs_dir):
            print(f"Valid model directory found: {model_dir}")
        else:
            # If either 'checkpoints' or 'validation_outputs' directory is missing or empty, remove the base directory
            print(f"Invalid model directory. Deleting: {model_dir}")
            shutil.rmtree(model_dir)

    print("Completed cleanup of model directories.")


if __name__ == "__main__":
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Cleanup model directories that do not contain valid checkpoints and validation outputs.')
    parser.add_argument('--base_dir', default='/is/cluster/fast/pghosh/ouputs/alpha_geo/cycle_gan/geometry/meta-llama/',
                        type=str, required=False, help='The base directory containing model directories to clean up.')

    # Parse arguments
    args = parser.parse_args()

    # Run the cleanup function
    cleanup_model_dirs(args.base_dir)
