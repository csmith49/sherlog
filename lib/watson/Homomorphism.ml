module IMap = CCMap.Make(Identifier)

type t = Term.t IMap.t

let empty = IMap.empty
let singleton = IMap.singleton
let of_list = IMap.of_list
let to_list = IMap.to_list

let rec apply h = function
    | Term.Variable x -> begin match IMap.find_opt x h with
        | Some term -> if Term.is_ground term then term else apply h term
        | None -> Term.Variable x end
    | Term.Function (f, args) -> 
        let args' = CCList.map (apply h) args in
            Term.Function (f, args')
    | (_ as term) -> term

let simplify h =
    let nontrivial key value = match value with
        | Term.Variable x -> not (Identifier.equal key x)
        | _ -> true
    in IMap.filter nontrivial h

let compose l r =
    let l = l |> IMap.map (apply r) |> simplify in
    let choose_left _ = function
        | `Left x -> Some x
        | `Right x -> Some x
        | `Both (x, _) -> Some x in
    IMap.merge_safe ~f:choose_left l r

module Unification = struct
    type equality = Eq of Term.t * Term.t

    let equate left right = Eq (left, right)

    let rec resolve_equalities eqs = resolve eqs empty
    (* simple implementation of Martelli-Montanari unification *)
    and resolve eqs h = match eqs with
        | [] -> Some h
        | Eq (x, y) :: rest when Term.equal x y -> resolve rest h
        | Eq (Term.Variable x, y) :: rest -> if Term.occurs x y then None else
            let h = compose h (singleton x y) in
            let rest = CCList.map
                (function Eq (l, r) -> Eq (apply h l, apply h r))
                rest in
            resolve rest h
        | Eq (x, Term.Variable y) :: rest -> if Term.occurs y x then None else
            let h = compose h (singleton y x) in
            let rest = CCList.map
                (function Eq (l, r) -> Eq (apply h l, apply h r))
                rest in
            resolve rest h
        | Eq (Term.Function (f, fargs), Term.Function (g, gargs)) :: rest ->
            if not (CCString.equal f g) then None else
            if not (CCInt.equal (CCList.length fargs) (CCList.length gargs)) then None else
                let argument_constraints = CCList.map2 equate fargs gargs in
                let constraints = argument_constraints @ rest in
                    resolve constraints h
        | _ -> None
end