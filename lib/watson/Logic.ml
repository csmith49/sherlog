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
end

module Obligation = struct
    open Language

    type t =
        | True
        | False
        | And of t list
        | Or of t list
        | Equal of ETerm.t * ETerm.t
        | Sample of ETerm.t * ETerm.t

    let rec variables = function
        | And os -> CCList.flat_map variables os
        | Or os -> CCList.flat_map variables os
        | Equal (l, r) -> (ETerm.variables l) @ (ETerm.variables r)
        | Sample (l, r) -> (ETerm.variables l) @ (ETerm.variables r)
        | _ -> []

    let rec map ob m = match ob with
        | And obs -> And (CCList.map (fun ob -> map ob m) obs)
        | Or obs -> Or (CCList.map (fun ob -> map ob m) obs)
        | Equal (l, r) -> Equal (ETerm.map l m, ETerm.map r m)
        | Sample (l, r) -> Sample (ETerm.map l m, ETerm.map r m)
        | _ -> ob

    let of_map m =
        let lift_pair (k, v) = Equal (ETerm.Variable k, ETerm.lift v) in
        let obs = m
            |> Term.Map.to_list
            |> CCList.map lift_pair in
        And obs
end