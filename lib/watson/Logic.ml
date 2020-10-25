module ETerm = struct
    open Language

    type t =
        | Variable of string
        | Constant of string
        | Integer of int
        | Boolean of bool
        | Float of float
        | Function of string * t list

    let lift = function
        | Term.Variable x -> Variable x
        | Term.Constant c -> Constant c
        | Term.Integer i -> Integer i
        | Term.Boolean b -> Boolean b
        | Term.Float f -> Float f

    let rec map t m = match t with
        | Variable x -> Term.Variable x
            |> (fun t -> Term.Map.map t m)
            |> lift
        | Function (f, fs) -> Function (f, CCList.map (fun t -> map t m) fs)
        | _ -> t

    let rec variables = function
        | Variable x -> [x]
        | Function (_, fs) -> CCList.flat_map variables fs
        | _ -> []

    let rec to_string = function
        | Variable x -> x
        | Constant c -> c
        | Integer i -> CCInt.to_string i
        | Float f -> CCFloat.to_string f
        | Boolean true -> "⟙" (* unicode large down-tack *)
        | Boolean false -> "⟘" (* unicode large up-tack *)
        | Function (f, fs) ->
            let fs' = fs
                |> CCList.map to_string
                |> CCString.concat ", " in
            f ^ "(" ^ fs' ^ ")"

    let equal l r = l == r
end

module Obligation = struct
    open Language

    type t =
        | True
        | False
        | And of t * t
        | Or of t * t
        | Equal of ETerm.t * ETerm.t
        | Sample of ETerm.t * ETerm.t

    let rec variables = function
        | And (l, r) -> CCList.flat_map variables [l ; r]
        | Or (l, r) -> CCList.flat_map variables [l ; r]
        | Equal (l, r) -> (ETerm.variables l) @ (ETerm.variables r)
        | Sample (l, r) -> (ETerm.variables l) @ (ETerm.variables r)
        | _ -> []

    let rec map ob m = match ob with
        | And (l, r) -> And (map l m, map r m)
        | Or (l, r) -> Or (map l m, map r m)
        | Equal (l, r) -> Equal (ETerm.map l m, ETerm.map r m)
        | Sample (l, r) -> Sample (ETerm.map l m, ETerm.map r m)
        | _ -> ob

    let conjoin obs = CCList.fold_left (fun l -> fun r -> And (l, r)) True obs
    let disjoin obs = CCList.fold_left (fun l -> fun r -> Or (l, r)) False obs

    let of_map m =
        let lift_pair (k, v) = Equal (ETerm.Variable k, ETerm.lift v) in
        let obs = m
            |> Term.Map.to_list
            |> CCList.map lift_pair in
        conjoin obs

    let rec to_string = function
        | True -> "⟙" (* unicode large down-tack *)
        | False -> "⟘" (* unicode large up-tack *)
        | And (l, r) ->
            let l' = to_string l in
            let r' = to_string r in
            l' ^ "∧" ^ r'
        | Or (l, r) ->
            let l' = to_string l in
            let r' = to_string r in
            l' ^ "∨" ^ r'
        | Equal (l, r) ->
            let l' = ETerm.to_string l in
            let r' = ETerm.to_string r in
            l' ^ " = " ^ r'
        | Sample (l, r) ->
            let l' = ETerm.to_string l in
            let r' = ETerm.to_string r in
            l' ^ " ~ " ^ r'

    (* right now, this does nothing at all *)
    let simplify = function
        | _ as obl -> obl
end