type t = [
    | `Rule of Watson.Rule.t
    | `Evidence of Evidence.t
    | `Parameter of Parameter.t
]

val encode_intro : 
    relation:string ->
    terms:Watson.Term.t list ->
    function_id:string ->
    arguments:Watson.Term.t list ->
    context:Watson.Term.t list ->
    body:Watson.Atom.t list -> t list

val encode_fuzzy :
    head:Watson.Atom.t ->
    body:Watson.Atom.t list ->
    weight:Watson.Term.t -> t list