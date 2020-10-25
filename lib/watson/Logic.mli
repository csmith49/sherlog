module ETerm : sig
    type t =
        | Variable of string
        | Constant of string
        | Integer of int
        | Boolean of bool
        | Float of float
        | Function of string * t list
    (** Existential terms for expresing the constraints on existential introduction *)

    val lift : Language.Term.t -> t
    (** [lift term] lifts a {!type:Term.t} object to an existential term *)

    val map : t -> Language.Term.Map.t -> t
    (** [map term m] applies the {!type:Term.Map.t} [m] to the existential term [term] *)

    val variables : t -> string list
    (** [variables term] returns a list of the string representations of all variables in [term] *)

    val to_string : t -> string

    val equal : t -> t -> bool
end

module Obligation : sig
    type t =
        | True
        | False
        | And of t * t
        | Or of t * t
        | Equal of ETerm.t * ETerm.t
        | Sample of ETerm.t * ETerm.t

    val variables : t -> string list

    val map : t -> Language.Term.Map.t -> t

    val of_map : Language.Term.Map.t -> t

    val to_string : t -> string

    val conjoin : t list -> t
    val disjoin : t list -> t

    val simplify : t -> t
end