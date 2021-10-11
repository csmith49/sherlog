type 'a t =
  | Identifier of string
  | Literal of 'a

let pp lit_pp ppf = function
    | Identifier id -> Fmt.pf ppf "%s" id
    | Literal lit -> Fmt.pf ppf "%a" lit_pp lit

module JSON = struct
    let encode encoder = function
        | Identifier id -> `Assoc [
                ("type", JSON.Encode.string "identifier");
                ("value", JSON.Encode.string id);
            ]
        | Literal v -> `Assoc [
                ("type", JSON.Encode.string "literal");
                ("value", encoder v);
        ]
end
