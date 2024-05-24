#!/home/mmordig/ont_project_all/ont_project_venv/bin/python
# #!/usr/bin/env python3
"""

WIP!!!

# Parses script.py for ##CONDOR lines, adds "script.exe" as executable, scriptArg1, scriptArg2 as arguments
condor_submit_bid arg1 arg2 --- script.exe scriptArg1 scriptArg2

condor_submit_bid 15 --submission-file submissionFile --- script.exe ddd
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
from typing import List

# common
logging.basicConfig(level=logging.INFO)

CONDOR_LINE_START = '##CONDOR'
CONDOR_SUBMIT_COMMAND = "condor_submit_bid"

def extract_condor_lines(input_file):
    """
    Extracts all lines starting with "##CONDOR" from the input file.
    """
    with open(input_file, 'r') as file:
        condor_lines = [line[len(CONDOR_LINE_START):].strip() for line in file if line.startswith(CONDOR_LINE_START)]
    return condor_lines

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

def check_is_executable(executable):
    """
    Check file can be executed
    """
    import distutils
    if not (os.access(executable, os.X_OK) or distutils.spawn.find_executable(executable)):
        logging.error(f"File '{executable}' is not executable or not found.")
        print(f"Do the following:\nchmod u+x '{executable}'")
        sys.exit(1)
    
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
        
def get_jobfile_name(jobfile_name, script_path):
    if jobfile_name is None:
        jobfile_name = Path(tempfile.mkdtemp(prefix="condor_job_"))
    if jobfile_name.is_dir():
        jobfile_name = create_job_script_filename(script_path=script_path, output_dir=jobfile_name)
    else:
        if jobfile_name.parent.exists():
            logging.error(f"Parent directory of jobfile {jobfile_name} does not exist.")
            sys.exit(1)
    return jobfile_name

# find last index of line of the form "Executable = something" with or without whitespace
def extract_last_toml_key(jobfile_lines, key):
    line_index = next((len(jobfile_lines)-1-i for i, line in enumerate(jobfile_lines[::-1]) if line.lower().startswith(key.lower())), None)
    if line_index is None:
        return None, None
    return line_index, jobfile_lines[line_index].split('=')[1].strip()

def rewrite_submission_file(jobfile_lines: List[str], extra_program_args):
    # rewrite jobfile (list of lines) with executable and arguments
    # always make sure that an executable is provided

    executable_line_index, executable = extract_last_toml_key(jobfile_lines, "Executable")
    arguments_line_index, arguments = extract_last_toml_key(jobfile_lines, "Arguments")
    logging.info(f"Found executable '{executable}' and arguments '{arguments}'")
    if arguments is None:
        arguments = []
    else:
        arguments = parse_htcondor_arguments(arguments)
    arguments.extend(extra_program_args)
    if executable is None:
        # make first argument the executable
        assert len(arguments) > 0, "must specify at least one argument to serve as executable"
        executable, arguments = arguments[0], arguments[1:]
    
    executable_line = f"Executable = {executable}"
    if executable_line_index is None:
        # insert before "queue"
        assert arguments_line_index is None, "You should move the first argument in the submission file to the executable line" # actually not a critical error, warning is sufficient
        jobfile_lines.insert(len(jobfile_lines)-1, executable_line)
    else:
        jobfile_lines[executable_line_index] = executable_line
    
    # quoting is not safe (but best approximation) because htcondor uses strange quoting rules
    arguments_line = f"Arguments = {format_htcondor_arguments(arguments)}" # quoting, e.g. '$(Cluster)' still expands the variables
    if arguments_line_index is None:
        # insert before "queue"
        jobfile_lines.insert(len(jobfile_lines)-1, arguments_line)
    else:
        jobfile_lines[arguments_line_index] = arguments_line
        
    return jobfile_lines

def run(args):
    script_and_args = args.script_and_args
    script_path = Path(script_and_args[0])
    if args.submission_file is not None:
        logging.info(f"Parsing submission info from '{args.submission_file}'")
        
        jobfile_lines = Path(args.submission_file).read_text().split("\n")
    else:
        logging.info(f"Parsing submission info from '{script_path}'")
        if not script_path.exists() or is_binary_file(script_path):
            raise ValueError("expected text file to parse submission info")
        jobfile_lines = extract_condor_lines(script_path)
        if not jobfile_lines:
            logging.warning(f"No lines starting with '{CONDOR_LINE_START}' found in the input bash script.")
            sys.exit(1)
    jobfile_lines = [x.strip() for x in jobfile_lines]
    jobfile_lines = [x for x in jobfile_lines if x] # remove empty lines
    # add queue if missing
    if not jobfile_lines[-1].startswith("queue"):
        jobfile_lines.append("queue")
    
    # logging.info("Got jobfile lines: " + str(jobfile_lines))
    logging.info(f"Adding extra args: {script_and_args}")
    rewrite_submission_file(jobfile_lines, script_and_args)
    # logging.info("Revised jobfile lines: " + str(jobfile_lines))
    
    jobfile_name = get_jobfile_name(jobfile_name=args.jobfile_name, script_path=script_path)
    logging.info(f"Writing job script to '{jobfile_name}'")
    create_condor_job_script(content="\n".join(jobfile_lines), job_script_filename=jobfile_name)
    print_job_script(job_script_filename=jobfile_name)
    
    logging.info(f"Submitting condor job with args: {args.condor_submit_args}")
    # args.dry = True # todo
    submit_condor_job(jobfile_name, condor_submit_args=args.condor_submit_args, dry_run=args.dry)

def yield_pairs(gen):
    x_prev = None
    i = -1
    for i, x in enumerate(gen):
        if i % 2 == 0:
            x_prev = x
        else:
            yield x_prev, x
    assert i % 2 == 1, "Odd number of elements"
    
def yield_occurrences(s, sub):
    i = 0
    while True:
        i = s.find(sub, i)
        if i == -1:
            break
        yield i
        i += 1

def parse_htcondor_arguments(arguments: str) -> List[str]:
    # https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html describes Arguments syntax
    # print(f"Arguments: {arguments}")
    if arguments.startswith('"'):
        # new syntax
        assert arguments.endswith('"'), f"Does not end with quote: '{arguments}'"
        arguments = arguments[1:-1].strip()
        # first, split by space(s) outside of quotes
        # arguments = arguments.split("'")
        
        # rule4
        replace_char = "\u0080"
        # replace_char = "|" # todo
        assert replace_char not in arguments, f"Replace char '{replace_char}' is in arguments"
        arguments = arguments.replace("''", replace_char)
        # now whenever it starts with ', we know it must start or end an argument
        
        # rule2
        old_arguments = arguments
        arguments = []
        # rule3: undo splitting by whitespace when single quotes appear
        prev_end_idx = -1
        for (start_idx, end_idx) in yield_pairs(yield_occurrences(old_arguments, "'")):
            arguments.extend(old_arguments[prev_end_idx+1:start_idx].split(" "))
            arguments.append(old_arguments[start_idx+1:end_idx])
            prev_end_idx = end_idx
        arguments.extend(old_arguments[prev_end_idx+1:].split(" "))
        
        arguments = [x for x in arguments if x]
        
        arguments = [x.replace('""', '"') for x in arguments] # rule1
        arguments = [x.replace(replace_char, "'") for x in arguments] # rule4
        return arguments
    else:
        # old syntax: argument separated by space(s)
        arguments = arguments.split(" ")
        arguments = [x for x in arguments if x] # remove empty strings
        arguments = [x.replace(r'\"', '"') for x in arguments] # replace \" with "
        return arguments
        
def format_htcondor_arguments(arguments: List[str], old_syntax: bool=False) -> str:
    # print(f"Arguments: {arguments}")
    # inverse of parse_htcondor_arguments
    if old_syntax:
        arguments = [x.replace('"', r'\"') for x in arguments] # replace " with \"
        return " ".join(arguments)
    else:
        arguments = [x.replace('"', '""') for x in arguments]
        arguments = [x.replace("'", "''") for x in arguments]
        add_quotes_if_whitespace = lambda s: f"'{s}'" if " " in s else s
        arguments = [add_quotes_if_whitespace(x) for x in arguments]
        arguments = " ".join(arguments)
        return f'"{arguments}"'

def main():
    parser = argparse.ArgumentParser(
        description=dedent(f"""\
            Submit jobs to {CONDOR_SUBMIT_COMMAND}, allowing extra arguments to be assigned automatically as executable and arguments.
            
            Not specifying an executable in the submission file (or with {CONDOR_LINE_START}) needs care.
            Run with --dry first to inspect.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('condor_submit_args', nargs='+', help=f'Additional arguments to {CONDOR_SUBMIT_COMMAND}, should include a bid')
    parser.add_argument('--jobfile-name', '-o', type=Path, help="Filename to save the Condor job script to; if it is a directory, "
                        "a suitably named file will be created in the directory; defaults to a temporary directory", default=None)
    parser.add_argument('--submission-file', '-s', type=Path, help="Submission file, defaults to first file", default=None)
    parser.add_argument('--dry', action='store_true', help="Don't actually submit the job, just print the job script")
    
    parser.add_argument('---', nargs=argparse.REMAINDER, dest="script_and_args", required=True, help='The path to the script to run with condor and its arguments')
    
    args = parser.parse_args()
    logging.info(f"Command line args are: {args}")
    run(args)
    
def tests():
    
    
    assert format_htcondor_arguments(["3", "simple", "arguments"], old_syntax=False) == '''"3 simple arguments"'''
    assert format_htcondor_arguments(["3", "simple", "arguments"], old_syntax=True) == '''3 simple arguments'''
    assert format_htcondor_arguments(["one", '"two"', "'three'"], old_syntax=True) == r"""one \"two\" 'three'"""
    assert format_htcondor_arguments(["one", "two with  spaces", "3", "'four'"], old_syntax=False) == '''"one 'two with  spaces' 3 ''four''"'''
    # not possible with old syntax
    assert format_htcondor_arguments(["one", '"two"', "spacey ' quoted' argument"], old_syntax=False) == '''"one ""two"" 'spacey '' quoted'' argument'"'''
    # not possible with old syntax

    # old syntax
    assert parse_htcondor_arguments(r"""one \"two\" 'three'""") == ['one', '"two"', "'three'"]
    # new syntax
    assert parse_htcondor_arguments('''"3 simple arguments"''') == ["3", "simple", "arguments"]
    assert parse_htcondor_arguments('''"one 'two with  spaces' 3 'four'"''') == ["one", "two with  spaces", "3", "four"]
    assert parse_htcondor_arguments('''"one ""two"" 'spacey '' quoted'' argument'"''') == ["one", '"two"', "spacey ' quoted' argument"]
    
    
    # will not be split correctly because $(Arguments) consists of more than one argument
    # """\
    #     AA = 1
    #     Arguments = aa bb
    #     Arguments = $(Arguments) cc
    # """
    
    for extra_lines in [[], ["extra_content"]]:
        # "Arguments", no extra args
        assert rewrite_submission_file(dedent("""\
            AA = 1
            Executable = cat
            Arguments = aa bb
        """).split("\n"), []) + extra_lines == dedent("""\
            AA = 1
            Executable = cat
            Arguments = "aa bb"
        """).split("\n") + extra_lines
        
        # "Arguments", extra args
        assert rewrite_submission_file(dedent("""\
            AA = 1
            Executable = cat
            Arguments = "aa bb"
        """).split("\n"), ["cc", "dd"]) + extra_lines == dedent("""\
            AA = 1
            Executable = cat
            Arguments = "aa bb cc dd"
        """).split("\n") + extra_lines
        
        # no "Arguments", no extra args
        assert rewrite_submission_file(dedent("""\
            AA = 1
            Executable = cat
        """).split("\n"), []) + extra_lines == dedent("""\
            AA = 1
            Executable = cat
            Arguments = ""
        """).split("\n") + extra_lines
        # todo: queue
        
        # no "Arguments", extra args
        assert rewrite_submission_file(dedent("""\
            AA = 1
            Executable = cat
        """).split("\n"), ["cc", "dd"]) + extra_lines == dedent("""\
            AA = 1
            Executable = cat
            Arguments = "cc dd"
        """).split("\n") + extra_lines
    
if __name__ == "__main__":
    # tests()
    main()
    
    # /home/mmordig/reinforcement/alphageometry/LLM_finetuner/condor_submit_with_extra_args.py 35 \
    #     --submission-file /home/mmordig/reinforcement/alphageometry/LLM_finetuner/echo_example.sub \
    #         --- arg1 arg2