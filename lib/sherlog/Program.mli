type t

val rules : t -> Watson.Rule.t list
val introduction_rules : t -> Watson.Rule.t list
val non_introduction_rules : t -> Watson.Rule.t list
val parameters : t -> Parameter.t list
val evidence : t -> Evidence.t list
val ontology : t -> Ontology.t

val make : Watson.Rule.t list -> Parameter.t list -> Evidence.t list -> Ontology.t -> t

module Filter : sig
    type t = Watson.Proof.t list -> Watson.Proof.t list
    
    val total : t
    (** [total proofs] is the identity filter that just returns [proofs] *)

    val constraint_avoiding : Ontology.t ->  t
    (** [constraint_avoiding ontology proofs] returns all proofs that avoid constraints given by [ontology] *)

    val intro_consistent : t
    (** [intro_consistent proofs] removes any proofs that are not introduction consistent *)
    
    val length : int -> t
    (** [length k proofs] removes proofs that are longer than [k] resolution steps *)

    val beam_width : Posterior.Score.t -> int -> t
    (** [beam_width score k proofs] samples [k] proofs from [proofs] proportional to [score] *)
    
    val uniform_width : int -> t
    (** [uniform_width k proofs] samples [k] proofs uniformly from [proofs] *)
    
    val compose : t -> t -> t
    (** [compose f g] constructs the filter where [f] is applied, then [g] *)

    val (>>) : t -> t -> t
    (** [f >> g] is short-hand for [compose f g] *)
end

val apply : t -> Watson.Proof.t -> Watson.Proof.t list
val prove : t -> Filter.t -> Watson.Atom.t list -> Watson.Proof.t list
val (|=) : t -> Watson.Atom.t list -> Watson.Proof.t list

val contradict : t -> Filter.t -> Watson.Proof.t -> Watson.Proof.t list

val models : t -> Filter.t -> Filter.t -> Watson.Atom.t list -> Model.t list

val pp : t Fmt.t

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end