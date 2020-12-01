module Term : sig
    type value =
        | Variable of string
        | Constant of string
        | Integer of int
        | Float of float
        | Boolean of bool
    and t =
        | Wildcard
        | Unit
        | Value of value
        | Pair of t * t

    val to_string : t -> string
    (** [to_string term] produces a human-readable string representation of [term] *)

    val compare : t -> t -> int
    (** [compare left right] compares [left] and [right] with an arbitrary, but consistent, total order *)

    val equal : t -> t -> bool
    (** [equal left right] checks if [left] and [right] are syntactically equivalent *)

    val variables : t -> string list
    (** [variables term] produces a list of all variables (as strings) in [term] *)

    val is_ground : t -> bool
    (** [is_ground term] returns [true] if [term] has no variables, and [false] otherwise *)

    val occurs : string -> t -> bool
    (** [occurs id term] returns [true] if [Variable id] is a sub-term of [term] and [false] otherwise *)

    val map : (value -> t) -> t -> t
end

module Map : sig
    type t
    (** a map transforms variables into terms *)

    val apply : t -> Term.t -> Term.t
    (** [apply map term] simultaneously maps all variables in [term] to their image in [map] *)

    val empty : t
    (** [empty] is a default map that acts as the identity when applied *)

    val singleton : string -> Term.t -> t
    (** [singleton id term] produces the map that maps [Variable id] to [term] and leaves every other variable unchanged *)

    val of_list : (string * Term.t) list -> t
    (** [of_list assoc] converts the association list [assoc] into a map *)

    val to_list : t -> (string * Term.t) list
    (** [to_list map] converts [map] to an association list *)

    val compose : t -> t -> t
    (** [compose left right] returns a map equivalent to applying [left] and then applying [right] *)

    module Unification : sig
        type equality
        (** equalities are unification constraints *)

        val equate : Term.t -> Term.t -> equality
        (** [equate left right] produces the unification constraint indicating we want a map [m] such that [m(left) == m(right)] *)

        val resolve_equalities : equality list -> t option
        (** [resolve_equalities eqs] resolves the list [eqs] and produces a map satisfying their indicated constraints, if able *)
    end
end

module Obligation : sig
    type t =
        | Assign of string
    
    val to_string : t -> string
    val applied_representation : Term.t list -> Term.t list -> t -> string

    val compare : t -> t -> int
    val equal : t -> t -> bool
end

module Atom : sig
    type t =
        | Atom of string * Term.t list
        | Intro of Obligation.t * Term.t list * Term.t list * Term.t list
    (** atoms are the fundamental relational values in Watson *)

    val to_string : t -> string
    (** [to_string atom] produces a human-readable string representation of [atom] *)

    val variables : t -> string list
    (** [variables atom] produces a list of all variables (as strings) contained in terms inside [atom] *)

    val unify : t -> t -> Map.t option
    (** [unify left right] produces the most-general map [m] such that [m(left) == m(right)], if such a map exists *)

    val apply : Map.t -> t -> t
    (** [apply map atom] applies [map] to all variables in terms inside [atom] *)

    val compare : t -> t -> int
    (** [compare left right] compares [left] and [right] with an arbitrary, but consistent, total order *)

    val equal : t -> t -> bool
    (** [equal left right] checks for syntactic equality between [left] and [right] *)
end