open Watson

type t = {
	rules : Rule.t list;
	parameters : Parameter.t list;
	evidence : Evidence.t list;
	ontology : Ontology.t;
}

let rules program = program.rules
let parameters program = program.parameters
let evidence program = program.evidence
let ontology program = program.ontology