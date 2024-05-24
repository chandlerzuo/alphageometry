from pathlib import Path
from setuptools import setup, find_packages

def strip_filter_comments(lines):
    lines = [x.strip() for x in lines]
    lines = [x for x in lines if not x.startswith("#") and len(x) > 0]
    # print(lines)
    return lines

setup(
    name='LLM_finetuner',
    version='0.1',
    # packages=[Path("LLM_finetuner") / dir for dir in find_packages("LLM_finetuner")], # so we can access files using "import LLM_finetuner.utils", otherwise "import utils"
    packages=["LLM_finetuner"],
    description='Fine tuning LLMs for alphageom',
    # long_description=open('README.md').read(),
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com/MyPackage',
    install_requires=strip_filter_comments(open('LLM_finetuner/requirements.txt').read().splitlines()),
)