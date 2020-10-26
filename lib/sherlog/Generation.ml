type t =
    | Gen of string * Watson.Syntax.Term.t list
    | Placeholder

let variables = function 
    | Gen (_, terms) -> CCList.flat_map Watson.Syntax.Term.variables terms
    | _ -> []