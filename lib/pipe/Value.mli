type 'a t =
    | Identifier of string
    | Literal of 'a

val pp : 'a Fmt.t -> 'a t Fmt.t

val equal : ('a -> 'a -> bool) -> 'a t -> 'a t -> bool

module JSON : sig
    val encode : 'a JSON.encoder -> 'a t JSON.encoder
end