type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
	| `Dependency of Watson.Rule.t
	| `Constraint of Watson.Atom.t list
]

val encode_intro : 
    relation:string ->
    arguments:Watson.Term.t list ->
    guard:string ->
    parameters:Watson.Term.t list ->
    context:Watson.Term.t list ->
    body:Watson.Atom.t list -> t list