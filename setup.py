from setuptools import setup, find_packages

setup(name='sherlog',
      version='0.9',
      description='Declarative, deep, probabilistic generative logic programming.',
      url='http://github.com/csmith49/sherlog',
      author='Calvin Smith',
      author_email='does@notexist.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          "networkx",
          "pyro-ppl",
          "torch",
          "rich",
          "click",
          "hashids",
          "pandas",
          "altair"
      ],
      zip_safe=False)
