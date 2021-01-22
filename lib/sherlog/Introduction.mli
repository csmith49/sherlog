type t = {
    f : string;
    args : Watson.Term.t list;
    context : Watson.Term.t list;
}

val placeholder : t

val variables : t -> string list

val to_json : t -> Yojson.Basic.t