type t

val rules : t -> Watson.Rule.t list
val parameters : t -> Parameter.t list
val evidence : t -> Evidence.t list
val ontology : t -> Ontology.t

val make : Watson.Rule.t list -> Parameter.t list -> Evidence.t list -> Ontology.t -> t

module Filter : sig
    type t = Watson.Proof.t list -> Watson.Proof.t list
    val total : t
    val intro_consistent : t
    val length : int -> t
    val width : int -> t
    val compose : t -> t -> t
    val (>>) : t -> t -> t
end

val apply : t -> Watson.Proof.t -> Watson.Proof.t list
val prove : t -> Filter.t -> Watson.Fact.t -> Watson.Proof.t list
val (|=) : t -> Watson.Fact.t -> Watson.Proof.t list

val contradict : t -> Filter.t -> Watson.Proof.t -> Watson.Proof.t list

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end