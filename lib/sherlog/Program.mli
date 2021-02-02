type t

val rules : t -> Watson.Rule.t list
val parameters : t -> Parameter.t list
val evidence : t -> Evidence.t list
val ontology : t -> Ontology.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end