type 'a t

module Functional : sig
    val target : 'a t -> string
    val function_id : 'a t -> string
    val arguments : 'a t -> 'a Value.t list

    val make : string -> string -> 'a Value.t list -> 'a t
end

module JSON : sig
    val encode : 'a JSON.encoder -> 'a t JSON.encoder
end