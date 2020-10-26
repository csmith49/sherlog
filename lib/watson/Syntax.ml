module Term = struct
    type t =
        | Variable of string
        | Constant of string
        | Integer of int
        | Float of float
        | Boolean of bool
        | Function of string * t list

    let rec to_string = function
        | Variable x -> x
        | Constant c -> c
        | Integer i -> CCInt.to_string i
        | Float f -> CCFloat.to_string f
        | Boolean true -> "⟙" (* unicode large down-tack *)
        | Boolean false -> "⟘" (* unicode large up-tack *)
        | Function (f, fs) ->
            let arguments = fs
                |> CCList.map to_string
                |> CCString.concat ", " in
            f ^ "(" ^ arguments ^ ")"

    let compare = Pervasives.compare
    let equal l r = (compare l r) == 0

    let rec is_ground = function
        | Variable _ -> false
        | Function (_, fs) -> CCList.for_all is_ground fs
        | _ -> true

    let rec occurs name term = match term with
        | Variable x -> CCString.equal name x
        | Function (_, fs) -> CCList.exists (occurs name) fs
        | _ -> false

    let rec variables = function
        | Variable x -> [x]
        | Function (_, fs) -> CCList.flat_map variables fs
        | _ -> []
end

module Map = struct
    module IdMap = CCMap.Make(CCString)
    type t = Term.t IdMap.t

    let rec apply map term = match term with
        | Term.Variable x -> begin match IdMap.find_opt x map with
            | Some term -> if Term.is_ground term then term else apply map term
            | None -> term end
        | Term.Function (f, fs) -> let fs = CCList.map (apply map) fs in
            Term.Function (f, fs)
        | _ -> term

    let empty = IdMap.empty
    let singleton = IdMap.singleton
    let of_list = IdMap.of_list
    let to_list = IdMap.to_list

    let simplify map =
        let nontrivial key value = match value with
            | Term.Variable x -> not (CCString.equal key x)
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
            | Eq (Variable x, y) :: rest -> if Term.occurs x y then None else
                let map = compose map (singleton x y) in
                let rest = CCList.map
                    (function Eq (l, r) -> Eq (apply map l, apply map r))
                    rest in
                resolve rest map
            | Eq (x, Variable y) :: rest -> if Term.occurs y x then None else
                let map = compose map (singleton y x) in
                let rest = CCList.map
                    (function Eq (l, r) -> Eq (apply map l, apply map r))
                    rest in
                resolve rest map
            | Eq (Function (f, fs), Function (g, gs)) :: rest when CCString.equal f g ->
                let sub_eqs = CCList.map2 equate fs gs in
                    resolve (rest @ sub_eqs) map
            | _ -> None
    end
end

module Obligation = struct
    type t =
        | Assign of string

    let to_string = function
        | Assign f -> f

    let compare = Pervasives.compare
    let equal left right = (compare left right) == 0
end

module Atom = struct
    type t =
        | Atom of string * Term.t list
        | Intro of Obligation.t * Term.t list * Term.t list * Term.t list

    let to_string = function
        | Atom (r, args) ->
            let args = args
                |> CCList.map Term.to_string
                |> CCString.concat ", " in
            r ^ "(" ^ args ^ ")"
        | Intro (obligation, parameters, context, values) ->
            let obligation = Obligation.to_string obligation in
            let args = (parameters @ context @ values)
                |> CCList.map Term.to_string
                |> CCString.concat ", " in
            "Intro(" ^ obligation ^ ", " ^ args ^ ")"

    let variables = function
        | Atom (_, args) -> CCList.flat_map Term.variables args
        | Intro (_, params, context, values) -> CCList.flat_map Term.variables (params @ context @ values)

    let apply map = function
        | Atom (r, args) ->
            let args = CCList.map (Map.apply map) args in
                Atom (r, args)
        | Intro (obligation, params, context, values) ->
            let params = CCList.map (Map.apply map) params in
            let context = CCList.map (Map.apply map) context in
            let values = CCList.map (Map.apply map) values in
                Intro (obligation, params, context, values)

    let unify left right = match left, right with
        | Atom (p, ps), Atom (q, qs) when CCString.equal p q ->
            let eqs = CCList.map2 Map.Unification.equate ps qs in
                Map.Unification.resolve_equalities eqs
        | Intro (po, pp, pc, pv), Intro (qo, qp, qc, qv) when Obligation.equal po qo ->
            let eqs = CCList.map2 Map.Unification.equate (pp @ pc @ pv) (qp @ qc @ qv) in
                Map.Unification.resolve_equalities eqs
        | _ -> None

    let compare = Pervasives.compare
    let equal l r = (compare l r) == 0
end