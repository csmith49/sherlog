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

    module M : sig
        val return : Watson.Proof.t -> Watson.Proof.t list
        val (>>=) : Watson.Proof.t list -> (Watson.Proof.t -> Watson.Proof.t list) -> Watson.Proof.t list
        
        val zero : Watson.Proof.t list
        val (++) : Watson.Proof.t list -> Watson.Proof.t list -> Watson.Proof.t list
    end

    (* construction *)
    val result : Watson.Proof.t list -> t
    val of_rule : Watson.Rule.t -> t

    module Combinator : sig
        (* disjunction *)
        val (<|>) : t -> t -> t
        val choice : t list -> t
        (* conjunction *)
        val (<&>) : t -> t -> t
        val union : t list -> t
        (* sequence *)
        val (>>) : t -> t -> t
        (* failure *)
        val attempt : t -> t
        (* recursion *)
        val fixpoint : t -> t
    end
end

module Filter : sig
    type t = Watson.Proof.t -> bool

    val negate : t -> t

    val constraint_avoiding : Ontology.t -> t

    val intro_consistent : t

    val length : int -> t

    val join : t -> t -> t

    val ( ++ ) : t -> t -> t
end

val apply : t -> Watson.Proof.t -> Watson.Proof.t list
val apply_with_dependencies : t -> Watson.Proof.t -> Watson.Proof.t list

val models : ?width:int -> t -> Posterior.t -> Watson.Atom.t list -> Model.t list