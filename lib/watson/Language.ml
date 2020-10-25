module Term = struct
    type t =
    | Variable of string
    | Constant of string
    | Integer of int
    | Float of float
    | Boolean of bool

    type term = t

    let to_string = function
        | Variable x -> x
        | Constant c -> c
        | Integer i -> CCInt.to_string i
        | Float f -> CCFloat.to_string f
        | Boolean true -> "⟙" (* unicode large down-tack *)
        | Boolean false -> "⟘" (* unicode large up-tack *)

    let equal left right = match left, right with
        | Variable x, Variable y -> CCString.equal x y
        | Constant a, Constant b -> CCString.equal a b
        | Integer i, Integer j -> CCInt.equal i j
        | Float f, Float g -> CCFloat.equal f g
        | Boolean p, Boolean q -> CCBool.equal p q
        | _, _ -> false

    let is_ground = function
        | Variable _ -> false
        | _ -> true

    let occurs name term = match term with
        | Variable x -> CCString.equal name x
        | _ -> false

    module Map = struct
        module StringMap = CCMap.Make(CCString)
        type t = term StringMap.t
        
        let rec map t m = match t with
            | Variable x -> begin match StringMap.find_opt x m with
                | Some t' -> if is_ground t' then t' else map t' m
                | None -> t end
            | _ -> t

        (* construction of maps relies on empty, singletons, and composition *)
        let empty = StringMap.empty
        let singleton x t = StringMap.singleton x t
        let of_list = StringMap.of_list

        (* removes trivial bindings to ensure application terminates *)
        let simplify m =
            let nontrivial key value = match value with
                | Variable x -> not (CCString.equal key x)
                | _ -> true
            in StringMap.filter nontrivial m

        (* composition applies left-then-right *)
        let compose left right =
            let left' = left
                |> StringMap.map (fun t -> map t right)
                |> simplify in
            let choose_left _ = function
                | `Left x -> Some x
                | `Right y -> Some y
                | `Both (x, _) -> Some x in
            StringMap.merge_safe ~f:choose_left left' right

        let to_list = StringMap.to_list
    end

    type equality = Equality of t * t

    let rec resolve_equalities eqs = resolve eqs Map.empty
    (* simple implementation of Martelli-Montanari unification *)
    and resolve eqs m = match eqs with
        | [] -> Some m
        | Equality (x, y) :: rest when equal x y -> resolve rest m
        | Equality (Variable x, y) :: rest -> if occurs x y then None else
            let m' = Map.compose m (Map.singleton x y) in
            let rest' = CCList.map
                (function Equality (l, r) -> Equality (Map.map l m', Map.map r m'))
                rest in
            resolve rest' m'
        | Equality (x, Variable y) :: rest -> if occurs y x then None else
            let m' = Map.compose m (Map.singleton y x) in
            let rest' = CCList.map
                (function Equality (l, r) -> Equality (Map.map l m', Map.map r m'))
                rest in
            resolve rest' m'
        | _ -> None
end

module Atom = struct
    type t = Atom of string * Term.t list

    let symbol = function Atom (f, _) -> f
    let arguments = function Atom (_, fs) -> fs
    let variables atom = atom |> arguments |> CCList.filter_map
        (fun t -> match t with
            | Term.Variable x -> Some x
            | _ -> None)
    let is_ground atom = atom |> arguments |> CCList.for_all Term.is_ground

    let compare left right = Pervasives.compare left right

    let equal left right = match left, right with
        | Atom (f, fs), Atom (g, gs) ->
            if CCString.equal f g then
                CCList.for_all2 Term.equal fs gs
            else false
    
    let to_string = function Atom (f, fs) ->
        let fs' = fs
            |> CCList.map Term.to_string
            |> CCString.concat ", " in
        f ^ "(" ^ fs' ^ ")"
    
    let unify = function Atom (f, fs) -> function Atom (g, gs) ->
        if CCString.equal f g then
            let eqs = CCList.map2 (fun l -> fun r -> Term.Equality (l, r)) fs gs in
            Term.resolve_equalities eqs
        else None
    
    let map atom m = match atom with
        | Atom (f, fs) ->
            let fs' = CCList.map (fun t -> Term.Map.map t m) fs in
            Atom (f, fs')

    module Position = struct
        type t = string * int

        let compare = Pervasives.compare
        let equal left right = (compare left right) == 0
    end

    let positions atom = atom
        |> arguments
        |> CCList.mapi (fun index -> function
            | Term.Variable x -> 
                let pos = (symbol atom, index) in
                Some (x, pos)
            | _ -> None)
        |> CCList.keep_some
end