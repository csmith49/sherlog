from subprocess import run
from generation import to_problog

PROGRAM = '/tmp/pli-program.pl'
EVIDENCE = '/tmp/pli-evidence.pl'

def lfi(problem, **kwargs):
    program, evidence = to_problog(problem)

    # write the files to a temporary location
    with open(PROGRAM, 'w') as f:
        f.write(program)
    
    with open(EVIDENCE, 'w') as f:
        f.write(evidence)

    # and call the external solver
    args = [
        "problog",
        "lfi",
        PROGRAM,
        EVIDENCE,
        "-k",
        "sdd"
    ]

    result = run(args, capture_output=True, text=True)

    return result.stdout