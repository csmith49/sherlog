type t
(** programs combine sets of rules with the extra structure needed to encode an optimization problem *)

(** {1 Access} *)

val rules : t -> Watson.Rule.t list
(** [rules program] returns a list of a rules comprising [program] *)

val introduction_rules : t -> Watson.Rule.t list
(** like [rules], but only returns rules whose conclusion is a value introduction *)

val non_introduction_rules : t -> Watson.Rule.t list
(** like [rules], but only returns rules with no value introductions  *)

val parameters : t -> Parameter.t list
(** [parameters program] returns a list of all learnable parameters defined by [program] *)

val evidence : t -> Evidence.t list
(** [evidence program] returns a list of all defined evidence in [program] *)

val ontology : t -> Ontology.t
(** [ontology program] returns the ontology defined in [program] *)

(** {1 Construction, Serialization, and Printing} *)

val make : Watson.Rule.t list -> Parameter.t list -> Evidence.t list -> Ontology.t -> t
(** [make] packages all the arguments into a program object *)

module JSON : sig
    (** Serialization done via the included [JSON] library *)
    
    val encode : t -> JSON.t
    (** [encode program] converts [program] to a JSON object *)

    val decode : JSON.t -> t option
    (** [decode json] attempts to extract a program from [json] *)
end

val pp : t Fmt.t
(** a pretty printer for program objects (TODO - needs breakpoint improvements) *)

module Semantics : sig
    type t = Watson.Proof.t -> Watson.Proof.t list

    val one : Watson.Rule.t list -> t
    (* applies the first possible rule in the list *)

    val all : Watson.Rule.t list -> t
    (* applies all possible rules in the list *)

    val fp : t -> t
    (* applies semantic transform until convergence *)

    val seq : t -> t -> t
    (* [seq l r] does [l] then [r] to all results *)

    val xor : t -> t -> t
    (* [xor l r] does [l] then [r] if [l] produces no results *)

    module Infix : sig
        val ( <+> ) : t -> t -> t
        (* xor *)
        
        val ( >> ) : t -> t -> t
        (* seq *)
    end
end

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


