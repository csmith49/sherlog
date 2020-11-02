from setuptools import setup

setup(name='sherlog',
      version='0.9',
      description='Declarative, deep, probabilistic generative logic programming.',
      url='http://github.com/csmith49/sherlog',
      author='Calvin Smith',
      author_email='does@notexist.com',
      license='MIT',
      packages=['sherlog'],
      install_requires=[
          "pyro",
          "torch"
      ],
      zip_safe=False)
