module Parameter : sig
    type domain =
        | Unit
        | Positive
        | Real

    val domain_to_string : domain -> string

    type t = Parameter of string * domain

    val to_string : t -> string
    val to_json : t -> Yojson.Basic.t
end

module Namespace : sig
    type t = Namespace of string

    val to_string : t -> string
    val to_json : t -> Yojson.Basic.t
end

type line = [
    | `Rule of Watson.Rule.t
    | `Query of Watson.Atom.t list
    | `Parameter of Parameter.t
    | `Namespace of Namespace.t
    | `Evidence of Watson.Atom.t list
]

val simplify_introduction : 
    relation:string ->
    arguments:Watson.Term.t list ->
    generator:string ->
    parameters:Watson.Term.t list ->
    context:Watson.Term.t list ->
    body:Watson.Atom.t list -> Watson.Rule.t list

type t = {
    parameters : Parameter.t list;
    namespaces : Namespace.t list;
    evidence : (Watson.Atom.t list) list;
    queries : (Watson.Atom.t list) list;
    program : Watson.Program.t;
}

val parameters : t -> Parameter.t list
val namespaces : t -> Namespace.t list
val evidence : t -> (Watson.Atom.t list) list
val queries : t -> (Watson.Atom.t list) list
val program : t -> Watson.Program.t

val of_lines : line list -> t
val to_string : t -> string
val to_json : t -> Yojson.Basic.t