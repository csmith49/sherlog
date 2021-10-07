type 'a t

module Functional : sig
    val statements : 'a t -> 'a Statement.t list

    val make : 'a Statement.t list -> 'a t
end

module JSON : sig
    val encode : 'a JSON.encoder -> 'a t JSON.encoder
end