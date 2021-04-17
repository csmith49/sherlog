from . import problog
from . import graph

default_parameterization = graph.Parameterization(
    stress=0.2,
    spontaneous=0.1,
    comorbid=0.3,
    influence=0.3
)