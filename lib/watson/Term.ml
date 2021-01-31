type t =
    | Variable of Identifier.t
    | Symbol of string
    | Integer of int
    | Float of float
    | Boolean of bool
    | Unit
    | Function of string * t list
    | Wildcard

let rec to_string = function
    | Variable x -> Identifier.to_string x
    | Symbol s -> s
    | Integer i -> CCInt.to_string i
    | Float f -> CCFloat.to_string f
    | Boolean true -> "T"
    | Boolean false -> "F"
    | Unit -> "()"
    | Function (f, args) ->
        let args' = args |> CCList.map to_string |> CCString.concat ", " in
        f ^ "(" ^ args' ^ ")"
    | Wildcard -> "_"

let compare l r = match l, r with
    | Wildcard, _ | _, Wildcard -> 0
    | _ -> Stdlib.compare l r
let equal l r = (compare l r) == 0

let rec is_ground = function
    | Variable _ -> false
    | Function (_, args) -> args |> CCList.for_all is_ground
    | _ -> true

let rec occurs x = function
    | Variable y when Identifier.equal x y -> true
    | Function (_, args) -> args |> CCList.exists (occurs x)
    | _ -> false

let rec variables = function
    | Variable x -> [x]
    | Function (_, args) -> args
        |> CCList.flat_map variables
        |> Identifier.uniq
    | _ -> []