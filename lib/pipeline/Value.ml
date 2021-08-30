type 'a t =
  | Identifier of string
  | Literal of 'a


let to_json literal_to_json = function
  | Identifier id ->
    `Assoc [
      ("type", `String "identifier");
      ("value", `String id);
    ]
  | Literal v ->
    `Assoc [
      ("type", `String "literal");
      ("value", literal_to_json v);
    ]
