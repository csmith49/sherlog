type t =
    | Gen of string * Watson.Term.t list
    | Placeholder

val variables : t -> string list

val to_json : t -> Yojson.Basic.t