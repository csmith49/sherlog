type t =
    | Gen of string * Watson.Term.t list
    | Placeholder

let variables = function 
    | Gen (_, terms) -> CCList.flat_map Watson.Term.variables terms
    | _ -> []

let to_json = function
    | Placeholder -> `Null
    | Gen (f, args) -> `Assoc [
        ("function", `String f);
        ("arguments", `List (CCList.map Utility.Term.to_json args));
    ]