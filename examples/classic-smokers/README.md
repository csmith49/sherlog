# Classic Smokers

## Unstated Dependencies

Generation depends on [NetworkX](https://networkx.org) for building scale-free graphs, and evaluation compares with [ProbLog](https://dtai.cs.kuleuven.be/problog/). The former is a dependency of Sherlog, but both can be installed with:
```
python3 -m pip install problog networkx
```

If you use a Mac, you will also need [PySDD](https://github.com/wannesm/PySDD). The installation may not be successful; if ProbLog complains that it can't find PySDD, find out what the problem is with
```
python3 -m pip install -vvv --upgrade --force-reinstall --no-deps --no-binary :all: pysdd
```