def to_sherlog(task):
    """Convert a social graph with concrete observations to a Sherlog problem.

    Parameters
    ----------
    task : Dict[str, Any]

    Returns
    -------
    str
    """
    with open("./programs/smokers.sl", "r") as f:
        source = f.read()
    
    graph_atoms, evidence = [], []

    # convert nodes
    for node in task["graph"]["nodes"]:
        graph_atoms.append(f"person(p_{node}).")
    
    # convert edges
    for (s, d) in task["graph"]["edges"]:
        graph_atoms.append(f"friend(p_{s}, p_{d}).")

    graph = '\n'.join(graph_atoms)

    # convert observations to evidence
    for observation in task["observations"]:
        atoms = []
        
        for node in observation["smokes"]["true"]:
            atoms.append(f"smokes(p_{node})")
        for node in observation["smokes"]["false"]:
            atoms.append(f"not_smokes(p_{node})")
        for node in observation["asthma"]["true"]:
            atoms.append(f"asthma(p_{node})")
        for node in observation["asthma"]["false"]:
            atoms.append(f"not_asthma(p_{node})")
        
        evidence.append(f"!evidence {', '.join(atoms)}.")
    
    return source + "\n\n" + graph + "\n\n" + '\n'.join(evidence)
