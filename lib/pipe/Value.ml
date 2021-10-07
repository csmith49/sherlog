type 'a t =
  | Identifier of string
  | Literal of 'a

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
