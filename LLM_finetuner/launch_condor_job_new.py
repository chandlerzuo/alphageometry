#!/home/mmordig/ont_project_all/ont_project_venv/bin/python
# #!/usr/bin/env python3

"""
Script that takes a script as an argument as well as a bid, takes all its lines starting with "##CONDOR" 
to create a condor job script and save it in a temporary directory with a useful name (basename of the file 
plus timestamp), prints it and then submits it with condor_submit.

Run with
python submit_condor_job.py <script> <bid>

Copyright Maximilian Mordig
"""

import logging
import os
import shlex
import sys
import argparse
import subprocess
import tempfile
from textwrap import dedent
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO)

CONDOR_LINE_START = '##CONDOR'
CONDOR_SUBMIT_COMMAND = "condor_submit_bid"
"""
The string that marks the start of a line that should be included in the condor job script.
"""

def extract_condor_lines(input_file):
    """
    Extracts all lines starting with "##CONDOR" from the input file.
    """
    with open(input_file, 'r') as file:
        condor_lines = [line[len(CONDOR_LINE_START):].strip() for line in file if line.startswith(CONDOR_LINE_START)]
    return condor_lines

def get_jobfile_lines_for_executable(script_and_args, wait_for_others: bool):
    """
    Get extra lines to add to the job file.
    """
    executable, *script_args = script_and_args
    
    if wait_for_others:
        script_args = (executable, tuple("$(Cluster) $(Process) $(NumRanks)".split(" "))) + script_args
        executable = os.path.expanduser("~/reinforcement/alphageometry/LLM_finetuner/distributed_tests/condor_wait_for_others.sh")
        # this script waits for the other jobs and sets useful environment variables
    
    # see here for double quotes for arguments: https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html#arguments
    # this may still not be safe
    # not safe:
    # Arguments = {shlex.join(script_args)}
    # when there is a string, it may be written with quotations 'str', so the argument becomes 'str', not str
    return dedent(f"""
    Executable = {executable}
    Arguments = "{shlex.join(script_args)}"
    """)
    
def get_extra_jobfile_content(script_and_args, time_limit, max_running_price, wait_for_others: bool):
    """
    Get lines to add to jobfile
    """
    
    res = ""
    if time_limit is not None:
        res += dedent(f"""\
            
            # hold after time limit is exceeded
            periodic_hold=(JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= {time_limit})
            periodic_hold_reason="Job runtime exceeded"
            
        """)
    if max_running_price > 0:
        res += dedent(f"""\
            
            # limit running cost
            +MaxRunningPrice=100
            +RunningPriceExceededAction="kill"
            
        """)
    res += get_jobfile_lines_for_executable(script_and_args, wait_for_others=wait_for_others)
    if "\nqueue" not in res:
        res += "\nqueue\n"
    else:
        assert not wait_for_others, "only launching 1 job, so does not make sense to wait for others"
        logging.info("Not adding 'queue' since already in file")
    return res
    
def create_job_script_filename(script_path, output_dir):
    """
    Creates a filename for the job script based on the script path, output_dir and a timestamp.
    """
    timestamp = int(time.time())
    job_name = Path(script_path).stem + f"_{timestamp}.sub"
    return Path(output_dir) / job_name
    
def create_condor_job_script(content, job_script_filename):
    """
    Write the job script to the given filename.
    """
    with open(job_script_filename, 'w') as file:
        file.write(content)

def print_job_script(job_script_filename):
    """
    Prints the job script to stdout.
    """
    print(f"Condor Job Submission Script ('{Path(job_script_filename).resolve()}'):")
    print("------------------")
    with open(job_script_filename, 'r') as file:
        print(file.read())
    print("------------------")

def get_submit_cmd(job_script_filename, condor_submit_args):
    """
    Returns the command to submit the job script with CONDOR_SUBMIT_COMMAND.
    """
    submit_cmd = [CONDOR_SUBMIT_COMMAND]
    submit_cmd.extend(condor_submit_args)
    # added into submission file instead
    # submit_cmd += ["-append", "+MaxRunningPrice=50", "-append", """+RunningPriceExceededAction="kill" """, "-append", "periodic_hold=(JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= 3600)", "-append", """periodic_hold_reason="Job runtime exceeded" """]
    submit_cmd.append(str(job_script_filename))
    return submit_cmd

def check_is_executable(filename):
    """
    Check file can be executed
    """
    if not os.access(filename, os.X_OK):
        logging.error(f"File '{filename}' is not executable.")
        print(f"Do the following:\nchmod u+x '{filename}'")
        sys.exit(1)
    
def submit_condor_job(job_script_filename, condor_submit_args, dry_run=False):
    """
    Submits the job script with CONDOR_SUBMIT_COMMAND.
    """
    submit_cmd = get_submit_cmd(job_script_filename, condor_submit_args)
    
    if dry_run:
        logging.warning(f"Dry run, would submit the job with the following command: {shlex.join(submit_cmd)}")
    else:
        # todo2: somehow not working, getting:
        # ERROR: Failed to commit job submission into the queue.
        # ERROR: The bid for this job is outside the valid margin.
        # replace current process with new process, make sure all file descriptors are flushed!
        # logging.shutdown()
        # os.execlp(*submit_cmd) # p to use PATH
        subprocess.run(submit_cmd, check=True)
    
def is_binary_file(file_path):
    """
    Heuristic to check whether a file is binary.
    
    Binary files often contain null bytes whereas text files often don't.
    It may fail for some legitimate text files.
    """
    with open(file_path, 'rb') as file:
        # Read the first few bytes (e.g., 1024 bytes) from the file
        data = file.read(1024)

    # Check for null bytes in the data
    return b'\x00' in data

def main(args):
    script_and_args = args.script_and_args
    script_path = Path(script_and_args[0])
    if not script_path.exists():
        logging.error(f"Script '{script_path}' does not exist.")
        sys.exit(1)
    if not (args.no_binary_check or not is_binary_file(script_path)):
        logging.error(f"Script '{script_path}' must be a text file to parse the condor options. Use '--no-binary-check' or submit it directly. Current working directory is '{os.getcwd()}'")
        sys.exit(1)
    
    # todo    
    # copy script to temporary directory
    # do not put to /tmp as this is not available on the remote machine
    
        
    check_is_executable(script_path)
    jobfile_lines = extract_condor_lines(script_path)
    if not jobfile_lines:
        logging.warning(f"No lines starting with '{CONDOR_LINE_START}' found in the input bash script.")
        sys.exit(1)

    jobfile_name = args.jobfile_name
    if args.jobfile_name is None:
        jobfile_name = Path(tempfile.mkdtemp(prefix="condor_job_"))
    if jobfile_name.is_dir():
        jobfile_name = create_job_script_filename(script_path=script_path, output_dir=jobfile_name)
    else:
        if jobfile_name.parent.exists():
            logging.error(f"Parent directory of jobfile {jobfile_name} does not exist.")
            sys.exit(1)
    
    jobfile_content = "# parameters extracted from comments\n" + "\n".join(jobfile_lines) + "\n" + get_extra_jobfile_content(script_and_args, time_limit=args.time_limit, max_running_price=args.max_running_price, wait_for_others=args.wait_for_others)
    jobfile_content += dedent(f"""\
    
    # Created on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
    # with command: 
    # {shlex.join(sys.argv)}
    # condor submit command is:
    # {shlex.join(get_submit_cmd(jobfile_name, condor_submit_args=args.condor_submit_args))}
    """)
    
    create_condor_job_script(content=jobfile_content, job_script_filename=jobfile_name)
    print_job_script(jobfile_name)

    submit_condor_job(jobfile_name, condor_submit_args=args.condor_submit_args, dry_run=args.dry)

if __name__ == "__main__":
    # script_basename = Path(sys.argv[0]).name
    script_basename = Path(__file__).name
    
    description = dedent(f"""
    Extract and submit Condor job from a script, e.g. Python, bash script etc. It also works with interactive scripts.
    
    Make sure this script is in your PATH by moving it into ~/.local/bin or with:
        export PATH="$PATH:{Path(__file__).resolve().parent}"
    
    Examples:
        {script_basename} 10 condorSubmitArg1 condorSubmitArg2 --jobfile-name "." --- myScript arg1 arg2
        {script_basename} 10 condorSubmitArg1 condorSubmitArg2 --jobfile-name "." --dry --- myScript arg1 arg2
        
        {script_basename} -h
        {script_basename} 10 --dry --- experimentation/launch_jupyter.sh --port 8888 --no-browser
        {script_basename} 10 -i --dry --- experimentation/launch_jupyter.sh --port 8888 --no-browser
        # request interactive input
        {script_basename} 10 --dry --- experimentation/example_request_input.py
    
    Incorrect usage:
        # uses a binary without any condor information
        {script_basename} 10 --dry --- echo H
    """)
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('condor_submit_args', nargs='+', help=f'Additional arguments to {CONDOR_SUBMIT_COMMAND}, should include a bid')
    parser.add_argument('--jobfile-name', '-o', type=Path, help="Filename to save the Condor job script to; if it is a directory, "
                        "a suitably named file will be created in the directory; defaults to a temporary directory", default=None)
    # parser.add_argument('--submission-file', '-s', type=Path, help="Submission file, defaults to first file", default=None)
    parser.add_argument('--dry', action='store_true', help="Don't actually submit the job, just print the job script")
    
    parser.add_argument('--time_limit', type=int, help="Time limit for job in seconds (int)", default=None)
    parser.add_argument('--max_running_price', type=int, help="Maximum running price (int), < 0 sets no limit", default=100)
    parser.add_argument('--wait_for_others', action='store_true', help="Wait for other jobs to start and set appropriate env variables (for distributed setup)", default=False)
    # parser.add_argument('--no-binary-check', action='store_true', help="Don't check whether the script is not a binary file", default=False)
     
     # nargs="*" does not parse arguments like "--port 8088" passed to script well since it thinks they are passed to this script
    parser.add_argument('---', nargs=argparse.REMAINDER, dest="script_and_args", required=True, help='The path to the script to run with condor and its arguments')
    # args = parser.parse_args()
    args, unknown_args = parser.parse_known_args()
    logging.debug(f"Known args are: {args}, unknown args are: {unknown_args}")
    args.condor_submit_args.extend(unknown_args) # these are passed to condor_submit, e.g. -i
    logging.info(f"Command line args are: {args}")
    
    main(args)