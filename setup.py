from setuptools import setup, find_packages

setup(name='sherlog',
    version='0.9',
    description='Declarative, deep, probabilistic generative logic programming.',
    url='http://github.com/csmith49/sherlog',
    author='Calvin Smith',
    author_email='email@cjsmith.io',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        "networkx",
        "matplotlib",
        "torch",
        "torchvision",
        "rich",
        "click",
        "hashids",
        "pandas",
        "altair",
        "altair_viewer"
    ],
    zip_safe=False,
    entry_points = {
        "console_scripts" : [
            "sherlog = sherlog.scripts.cli:cli"
        ]
    })