type t =
    | Variable of string
    | Symbol of string
    | Integer of int
    | Float of float
    | Boolean of bool
    | Function of string * t list
    | Unit

let rec hash = function
    (* use a pair hash w/ vars and symbols to disambiguate *)
    | Variable x -> CCHash.(pair string string) ("variable", x)
    | Symbol s -> CCHash.(pair string string) ("symbol", s)
    | Integer i -> CCHash.int i
    | Float f -> CCFloat.hash f (* for some reason this isn't in CCHash? *)
    | Boolean b -> CCHash.bool b
    | Function (f, args) -> CCHash.(pair string (list hash)) (f, args)
    | Unit -> 13 (* might as well pick a constant *)

let compare = Stdlib.compare
let equal l r = (compare l r ) == 0

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
    | Function (f, args) ->
        pf ppf "%s(@[<1>%a@])" f (list ~sep:comma pp) args

let to_string = Fmt.to_to_string pp

module Make = struct
    module Shorthand = struct
        let v x = Variable x
        let s x = Symbol x
        let i x = Integer x
        let r x = Float x
        let b x = Boolean x
        let f x xargs = Function (x, xargs)
        let u = Unit
    end

    module Variable = struct
        let tagged x i =
            let name = x ^ ":" ^ (CCInt.to_string i) in
            Variable name

        let rec avoiding variables =
            avoiding_aux variables 0
        and avoiding_aux variables counter =
            let name = avoiding_name counter in
            if CCList.mem ~eq:CCString.equal name variables then
                Variable name
            else
                avoiding_aux variables (counter + 1)
        and avoiding_name counter = "avoiding:" ^ (CCInt.to_string counter) 

        let wildcard_counter = ref 0
        let wildcard = function () ->
            let counter = !wildcard_counter in
            let _ = wildcard_counter := !wildcard_counter + 1 in
            tagged "wildcard" counter
    end
end

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
        | _ -> None
end
