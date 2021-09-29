(*  *)
type t

(* Construction *)
val empty : t
val singleton : string -> Term.t -> t

(* Conversion *)
val of_assoc : (string * Term.t) list -> t
val to_assoc : t -> (string * Term.t) list

(* Utility *)

(* apply sub to term *)
val apply : t -> Term.t -> Term.t

(* simplify sub *)
val simplify : t -> t

(* join two subs together *)
val compose : t -> t -> t

(* convert to string *)
val to_string : t -> string

(*  *)
module Unification : sig
    (*  *)
    type equality

    (*  *)
    val equate : Term.t -> Term.t -> equality

    (*  *)
    val resolve : equality list -> t option
end

module Generalization : sig
    type generalization

    val generalize : Term.t -> Term.t -> generalization

    val resolve : generalization list -> t option
end

(*  *)
module JSON : sig
    (*  *)
    val encode : t -> JSON.t
    (*  *)
    val decode: JSON.t -> t option
end


(* Infix utilities *)
module Infix : sig
    (* alias for Unification.equate *)
    val (=?=) : Term.t -> Term.t -> Unification.equality

    (* alias for apply *)
    val ($) : t -> Term.t -> Term.t

    (* alias for compose *)
    val (>->) : t -> t -> t
end