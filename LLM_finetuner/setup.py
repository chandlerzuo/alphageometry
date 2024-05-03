from setuptools import setup, find_packages

setup(
    name='LLM_finetuner',
    version='0.1',
    packages=find_packages("."),
    description='Fine tuning LLMs for alphageom',
    # long_description=open('README.md').read(),
    author='Your Name',
    author_email='your.email@example.com',
    url='http://example.com/MyPackage',
    install_requires=open('requirements.txt').read().splitlines(),
)