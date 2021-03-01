type t =
    | Variable of string
    | Symbol of string
    | Integer of int
    | Float of float
    | Boolean of bool
    | Unit
    | Function of string * t list
    | Wildcard

let compare l r = match l, r with
    | Wildcard, _ | _, Wildcard -> 0
    | _ -> Stdlib.compare l r
let equal l r = (compare l r) == 0

let rec is_ground = function
    | Variable _ -> false
    | Function (_, args) -> args |> CCList.for_all is_ground
    | _ -> true

let rec occurs x = function
    | Variable y when CCString.equal x y -> true
    | Function (_, args) -> args |> CCList.exists (occurs x)
    | _ -> false

let rec variables = function
    | Variable x -> [x]
    | Function (_, args) -> args
        |> CCList.flat_map variables
        |> CCList.uniq ~eq:CCString.equal
    | _ -> []

let rec pp ppf term = let open Fmt in match term with
    | Variable x -> (styled (`Fg `Red) string) ppf x
    | Symbol s -> (styled (`Fg `Yellow) string) ppf s
    | Integer i -> (styled (`Fg `Green) int) ppf i
    | Float f -> (styled (`Fg `Green) float) ppf f
    | Boolean b -> (styled (`Fg `Green) bool) ppf b
    | Unit -> pf ppf "()"
    | Wildcard -> (styled (`Faint) string) ppf "_"
    | Function (f, args) ->
        pf ppf "%s(@[<1>%a@])" f (list ~sep:comma pp) args

let to_string = Fmt.to_to_string pp

module JSON = struct
    let lift typ encoder value = `Assoc [
        ("type", `String typ);
        ("value", encoder value);
    ]
    let rec encode = function
        | Variable x -> lift "variable" JSON.Make.string x
        | Symbol s -> lift "symbol" JSON.Make.string s
        | Integer i -> lift "integer" JSON.Make.int i
        | Float f -> lift "float" JSON.Make.float f
        | Boolean b -> lift "boolean" JSON.Make.bool b
        | Unit -> `Assoc [("type", `String "unit")]
        | Function (f, args) -> `Assoc [
            ("type", `String "function");
            ("value", `String f);
            ("arguments", `List (args |> CCList.map encode));
        ]
        | Wildcard -> `Assoc [("type", `String "wildcard")]

    let rec decode json = match JSON.Parse.(find string "type" json) with
        | Some "variable" -> json
            |> JSON.Parse.(find string "value")
            |> CCOpt.map (fun x -> Variable x)
        | Some "symbol" -> json
            |> JSON.Parse.(find string "value")
            |> CCOpt.map (fun s -> Symbol s)
        | Some "integer" -> json
            |> JSON.Parse.(find int "value")
            |> CCOpt.map (fun i -> Integer i)
        | Some "float" -> json
            |> JSON.Parse.(find float "value")
            |> CCOpt.map (fun f -> Float f)
        | Some "boolean" -> json
            |> JSON.Parse.(find bool "value")
            |> CCOpt.map (fun b -> Boolean b)
        | Some "unit" -> Some Unit
        | Some "function" ->
            let f = json |> JSON.Parse.(find string "value") in
            let args = json
                |> JSON.Parse.(find (list decode) "arguments") in
            CCOpt.map2 (fun f -> fun args -> Function (f, args)) f args
        | Some "wildcard" -> Some Wildcard
        | _ -> None
end