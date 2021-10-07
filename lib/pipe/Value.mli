type 'a t =
    | Identifier of string
    | Literal of 'a

module JSON : sig
    val encode : 'a JSON.encoder -> 'a t JSON.encoder
end