module Term = struct
    type value =
        | Variable of string
        | Constant of string
        | Integer of int
        | Float of float
        | Boolean of bool
    and t =
        | Wildcard
        | Unit
        | Value of value
        | Pair of t * t

    let rec to_string = function
        | Value (Variable x) -> x
        | Value (Constant c) -> c
        | Value (Integer i) -> CCInt.to_string i
        | Value (Float f) -> CCFloat.to_string f
        | Value (Boolean true) -> "⟙" (* unicode large down-tack *)
        | Value (Boolean false) -> "⟘" (* unicode large up-tack *)
        | Unit -> "()"
        | Wildcard -> "_"
        | Pair (l, r) ->
            let l = to_string l in
            let r = to_string r in
                "(" ^ l ^ " :: " ^ r ^ ")"

    let compare l r = match l, r with
        | Wildcard, _ | _, Wildcard -> 0
        | _ -> Stdlib.compare l r
    let equal l r = (compare l r) == 0

    let rec map f = function
        | Value v -> f v
        | Pair (l, r) -> Pair (map f l, map f r)
        | (_ as term) -> term

    let rec is_ground = function
        | Value (Variable _) -> false
        | Pair (l, r) -> (is_ground l) && (is_ground r)
        | _ -> true

    let rec occurs name = function
        | Value (Variable x) -> CCString.equal name x
        | Pair (l, r) -> (occurs name l) || (occurs name r)
        | _ -> false

    let rec variables = function
        | Value (Variable x) -> [x]
        | Pair (l, r) -> (variables l) @ (variables r)
        | _ -> []
end

module Map = struct
    module IdMap = CCMap.Make(CCString)
    type t = Term.t IdMap.t

    let rec apply map term =
        let f value = match value with
            | Term.Variable x -> begin match IdMap.find_opt x map with
                | Some term -> if Term.is_ground term then term else apply map term
                | None -> Term.Value value end 
            | _ -> Term.Value value in
        Term.map f term

    let empty = IdMap.empty
    let singleton = IdMap.singleton
    let of_list = IdMap.of_list
    let to_list = IdMap.to_list

    let simplify map =
        let nontrivial key value = match value with
            | Term.Value Term.Variable x -> not (CCString.equal key x)
            | _ -> true
        in IdMap.filter nontrivial map

    let compose left right =
        let left = left
            |> IdMap.map (apply right)
            |> simplify in
        let choose_left _ = function
            | `Left x -> Some x
            | `Right x -> Some x
            | `Both (x, _) -> Some x in
        IdMap.merge_safe ~f:choose_left left right

    module Unification = struct
        type equality = Eq of Term.t * Term.t

        let equate left right = Eq (left, right)

        let rec resolve_equalities eqs = resolve eqs empty
        (* simple implementation of Martelli-Montanari unification *)
        and resolve eqs map = match eqs with
            | [] -> Some map
            | Eq (x, y) :: rest when Term.equal x y -> resolve rest map
            | Eq (Term.Value Variable x, y) :: rest -> if Term.occurs x y then None else
                let map = compose map (singleton x y) in
                let rest = CCList.map
                    (function Eq (l, r) -> Eq (apply map l, apply map r))
                    rest in
                resolve rest map
            | Eq (x, Term.Value Variable y) :: rest -> if Term.occurs y x then None else
                let map = compose map (singleton y x) in
                let rest = CCList.map
                    (function Eq (l, r) -> Eq (apply map l, apply map r))
                    rest in
                resolve rest map
            | Eq (Pair (l1, l2), Pair (r1, r2)) :: rest ->
                let constraints = Eq (l1, r1) :: Eq (l2, r2) :: rest in
                    resolve constraints map
            | _ -> None
    end
end

module Guard = struct
    type t = string

    let to_string g = g

    let applied_representation target parameters guard =
        let target = Term.to_string target in
        let parameters = parameters
            |> CCList.map Term.to_string
            |> CCString.concat ", " in
        target ^ " = " ^ guard ^ "[" ^ parameters ^ "]"

    let compare = CCString.compare
    let equal left right = (compare left right) == 0
end

module Atom = struct
    type t =
        | Atom of string * Term.t list
        | Intro of Guard.t * Term.t list * Term.t list * Term.t

    let subterms = function
        | Atom (_, args) -> args
        | Intro (_, parameters, context, target) -> parameters @ context @ [target]

    let to_string = function
        | Atom (r, args) ->
            let args = args
                |> CCList.map Term.to_string
                |> CCString.concat ", " in
            r ^ "(" ^ args ^ ")"
        | Intro (guard, _, _, _) as intro ->
            let guard = Guard.to_string guard in
            let args = subterms intro
                |> CCList.map Term.to_string
                |> CCString.concat ", " in
            "Intro(" ^ guard ^ ", " ^ args ^ ")"

    let variables atom = atom
        |> subterms
        |> CCList.flat_map Term.variables

    let apply map = function
        | Atom (r, args) ->
            let args = CCList.map (Map.apply map) args in
                Atom (r, args)
        | Intro (guard, params, context, target) ->
            let params = CCList.map (Map.apply map) params in
            let context = CCList.map (Map.apply map) context in
            let target = Map.apply map target in
                Intro (guard, params, context, target)

    let unify left right = match left, right with
        | Atom (p, _), Atom (q, _) when CCString.equal p q ->
            let eqs = CCList.map2 Map.Unification.equate (subterms left) (subterms right) in
                Map.Unification.resolve_equalities eqs
        | Intro (p, _, _, _), Intro (q, _, _, _) when Guard.equal p q ->
            let eqs = CCList.map2 Map.Unification.equate (subterms left) (subterms right) in
                Map.Unification.resolve_equalities eqs
        | _ -> None

    let compare = Stdlib.compare
    let equal l r = (compare l r) == 0
end
