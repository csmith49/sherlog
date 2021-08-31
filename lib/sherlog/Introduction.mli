(* Introduction Schema *)

(* Each attribute of an introduction is equipped with `key`, `to_term`, and `of_term` to support
   embedding into Watson terms. *)

module type ATTRIBUTE_EMBEDDING = sig
    type attribute

    val key : string
    val to_term : attribute -> Watson.Term.t
    val of_term : Watson.Term.t -> attribute option
end

module Target : sig
    type t = Watson.Term.t
    (* the target of the intro - either a value or constant *)
    
    include (ATTRIBUTE_EMBEDDING with type attribute := t)
end

module FunctionID : sig
    type t = string
    (* function used to introduce target *)
    
    include (ATTRIBUTE_EMBEDDING with type attribute := t)
end

module Arguments : sig
    type t = Watson.Term.t list
    (* arguments passed to function *)
    
    include (ATTRIBUTE_EMBEDDING with type attribute := t)
end

module Context : sig
    type t = Watson.Term.t list
    (* context in which the function is applied *)
    
    include (ATTRIBUTE_EMBEDDING with type attribute := t)
end

(* Introductions *)

type t = {
    target : Target.t;
    function_id : FunctionID.t;
    arguments : Arguments.t;
    context : Context.t;
}
(* target <- function_id(arguments) @ context *)

(* Functional interface *)

val make : Target.t -> FunctionID.t -> Arguments.t -> Context.t -> t

val target : t -> Target.t
val function_id : t -> FunctionID.t
val arguments : t -> Arguments.t
val context : t -> Context.t

(* Atom Embedding *)

val relation : string
(* relation used for embedding atom *)

val to_atom : t -> Watson.Atom.t
val of_atom : Watson.Atom.t -> t option

(* Utilities *)

val targetless_identifier : t -> string
(* uniquely identifies introduction source *)

val is_constrained : t -> bool
(* true if the intro's target is a variable, false otherwise *)

val introduction_consistency_tag : t -> Watson.Term.t list