module Term : sig
    (** {1 Term} *)

    (** Values and variables are represented in Watson by terms. As Watson does not allow function symbols, the term representation is flat, and many of the utility functions provided are fairly trivial cases of pattern-matching. *)

    type t =
        | Variable of string
        | Constant of string
        | Integer of int
        | Float of float
        | Boolean of bool

    type term = t
    (** {!type:term} is a simple type alias for {!type:t} *)

    val equal : t -> t -> bool
    (** [equal l r] checks for semantic equality of [l] and [r] *)

    val is_ground : t -> bool
    (** [is_ground term] returns [true] if [term] is not a variable, and [false] otherwise *)

    val to_string : t -> string
    (** [to_string term] uses built-in [to_string] functions to convert [term] to a string *)

    (** {2 Map} *)

    (** In Watson, a map is a shorter name for a term substitution. Maps are exclusively constructed via unification, which is achieved at the term level by constructing {!type:equality} constraints and resolving them with {!value:resolve_equalities}. *)

    module Map : sig
        type t
        (** A map transforms variables (possibly embedded in other objects) into terms *)

        val map : term -> t -> term
        (** [map term m] applies the map [m] to [term] *)

        val compose : t -> t -> t
        (** [compose l r] constructs a new map that is equivalent to [r(l(term))] *)

        val empty : t

        val of_list : (string * term) list -> t

        val to_list : t -> (string * term) list
        (** [to_list m] converts [m] to an association list *)
    end

    type equality = Equality of t * t
    (** The value [Equality (l, r)] represents the constraint that [l] and [r] should unify to the same value *)

    val resolve_equalities : equality list -> Map.t option
    (** [resolve_equalities eqs] produces the most-general unifier (a {!type:Map.t} object) if one exists, and returns [None] otherwise *)
end

module Atom : sig
    (** {1 Atom} *)

    (** Atoms are the fundamental relational values in Watson. *)

    type t = Atom of string * Term.t list
    (** [Atom (f, fs)] is the value constructed using relation [f] and arguments [fs] *)

    val symbol : t -> string
    (** [symbol (Atom (f, _))] returns the symbol [f] *)

    val arguments : t -> Term.t list
    (** [symbol (Atom (_, fs))] returns the arguments [fs] *)

    val variables : t -> string list
    (** [variables atom] returns a list of all variables (as strings) in the arguments of [atom] *)

    val is_ground : t -> bool
    (** [is_ground atom] returns [true] if [atom] contains no variables, and [false] otherwise *)

    val equal : t -> t -> bool
    (** [equal l r] tests for syntactic equality between [l] and [r] *)

    val compare : t -> t -> int

    val to_string : t -> string
    (** [to_string atom] converts [atom] to a human-readable string format *)

    val unify : t -> t -> Term.Map.t option
    (** [unify l r] returns a {!type:Term.Map.t} representing the most-general unifier of [l] and [r], if one exists *)

    val map : t -> Term.Map.t -> t
    (** [map atom m] applies map [m] to all arguments of [atom] *)

    module Position : sig
        type t
        (** A position marks an index in a particular relation *)

        val compare : t -> t -> int
        (** A lexicographic comparison of positions *)

        val equal : t -> t -> bool
        (** [equal l r] tests for syntactic equality between [l] and [r] *)
    end

    val positions : t -> (string * Position.t) list
    (** [positions atom] returns an association list of all variables (as strings) in [atom] and their positions *)
end