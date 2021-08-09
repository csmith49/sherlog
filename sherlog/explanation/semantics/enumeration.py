from typing import Callable, List, Optional
from torch import Tensor, zeros, tensor
from torch.distributions import Categorical, Bernoulli
from itertools import chain
from collections import defaultdict

from ...engine import Functor, Assignment
from ...logs import get

logger = get("story.semantics.enumeration")

# goal: dummy functor that applies enumerators to assignments and collects result
Enumerator = Callable[[Assignment], Optional[List[Tensor]]]

# builtin enumerators
def categorical_enumerator(assignment : Assignment) -> Optional[List[Tensor]]:
    # find number of parameters
    dimension = len(list(assignment.dependencies))

    # build dummy dist
    dummy_logits = zeros(dimension)
    dist = Categorical(logits=dummy_logits)

    # return enumerated values
    return dist.enumerate_support().unbind()

def bernoulli_enumerator(assignment : Assignment) -> Optional[List[Tensor]]:
    # build dummy dist
    dummy_logit = tensor(0.0)
    dist = Bernoulli(logits=dummy_logit)

    return dist.enumerate_support().unbind()

# functor constructors
def wrap(obj, **kwargs):
    return []

def fmap(callable, args, kwargs, **fmap_args):
    return list(chain(**args))

def builtin_factory(enumerator=lambda _: None):
    def builtin(*args, **kwargs):
        # unpack the assignment and apply the pred
        assignment = kwargs['assignment']
        domain = enumerator(assignment)
        target = assignment.target

        # if the enumerator gives useful info, short-circuit
        if domain:
            result = [(target, domain)]
            return list(chain(result, *args))
        
        # if we never short-circuit, default to no enumeration
        return list(chain(*args))
    return builtin

# functor factory
def factory(enumerators={
    'categorical': categorical_enumerator,
    'bernoulli' : bernoulli_enumerator
}):
    lifted_builtins = defaultdict(lambda: builtin_factory())
    
    # register the builtins to be checked by the predicates
    for builtin, enumerator in enumerators.items():
        lifted_builtins[builtin] = builtin_factory(enumerator=enumerator)
    
    return Functor(wrap, fmap, lifted_builtins)