(* Value *)

module Value : sig
    type t

    val of_term : Watson.Term.t -> t option

    val pp : t Fmt.t

    module JSON : sig
        val encode : t -> JSON.t
    end
end

(* Statement *)

module Statement : sig
    type t

    val of_introduction : Introduction.t -> t option

    val make : string -> string -> Watson.Term.t list -> t option

    val pp : t Fmt.t

    val equal : t -> t -> bool

    module JSON : sig
        val encode : t -> JSON.t
    end
end

(* Model *)

type t

val of_statements : Statement.t list -> t

val pp : t Fmt.t

module JSON : sig
    val encode : t -> JSON.t
end