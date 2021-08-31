module Value : sig
    type 'a t =
        | Identifier of string
        | Literal of 'a

    val to_json : ('a -> JSON.t) -> 'a t -> JSON.t
end

module Statement : sig
    type 'a t = {
        target : string;
        function_id : string;
        arguments : 'a Value.t list;
    }

    val to_json : ('a -> JSON.t) -> 'a t -> JSON.t
end

type 'a t = 'a Statement.t list

val to_json : ('a -> JSON.t) -> 'a t -> JSON.t