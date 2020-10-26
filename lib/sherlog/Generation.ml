type t =
    | Gen of string * Watson.Syntax.Term.t list
    | Placeholder

let variables = function 
    | Gen (_, terms) -> CCList.flat_map Watson.Syntax.Term.variables terms
    | _ -> []

let to_json = function
    | Placeholder -> `Null
    | Gen (f, args) -> `Assoc [
        ("function", `String f);
        ("arguments", `List (CCList.map Program.term_to_json args));
    ]