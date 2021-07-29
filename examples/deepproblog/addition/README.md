# Forth Addition

Two neural predicates --- `result/4` and `carry/4` --- used in the `slot/4` predicate. The first three arguments are the two digits and the previous carry to be summed. The next two arguments are the new carry and the new resulting digit.

The target predicate `add/5` has the following arguments: the two list of input digits, the input carry, the resulting carry, and the resulting sum.