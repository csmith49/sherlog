module IMap = CCMap.Make(CCString)

type t = Term.t IMap.t

let empty = IMap.empty
let singleton = IMap.singleton
let of_assoc = IMap.of_list
let to_assoc = IMap.to_list

let rec apply h = function
    | Term.Variable x -> begin match IMap.find_opt x h with
        | Some (Term.Variable y as term) when CCString.equal x y -> term
        | Some term -> if Term.is_ground term then term else apply h term
        | None -> Term.Variable x end
    | Term.Function (f, args) -> let args' = CCList.map (apply h) args in
        Term.Function (f, args')
    | (_ as term) -> term

let simplify h =
    let nontrivial key value = match value with
        | Term.Variable x -> not (CCString.equal key x)
        | _ -> true
    in IMap.filter nontrivial h

let compose l r =
    let l = l |> IMap.map (apply r) |> simplify in
    let choose_left _ = function
        | `Left x -> Some x
        | `Right x -> Some x
        | `Both (x, _) -> Some x in
    IMap.merge_safe ~f:choose_left l r

let to_string h = h
    |> IMap.to_list
    |> CCList.map (fun (k, v) ->
            k ^ " : " ^ (Term.to_string v)
        )
    |> CCString.concat ", "
    |> fun s -> "[" ^ s ^ "]"

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

module GeneralizationLattice = struct
    (* definitional *)
    
    type gen_c = Gen of Term.t * Term.t
    let gen_mk l r = Gen (l, r)
    let gen_map f = function Gen (x, y) -> Gen (f x, f y)
    
    (** [generalize large small] returns a substitution [h] such that [h $ large == small], if one exists *)
    let rec generalize large small =
        generalize_aux [ Gen (large, small) ] empty
    (* one-sided version of Martelli-Montanari unification *)
    and generalize_aux eqs h = match eqs with
        | [] -> Some h
        | Gen (x, y) :: rest when Term.equal x y -> generalize_aux rest h
        | Gen (Term.Variable x, y) :: rest ->
            if Term.occurs x y then None else
            let h = compose h (singleton x  y) in
            let rest = CCList.map (gen_map (apply h)) rest in
                generalize_aux rest h
        | Gen (Term.Function (f, fargs), Term.Function (g, gargs)) :: rest ->
            if not (CCString.equal f g) then None else
            if not (CCInt.equal (CCList.length fargs) (CCList.length gargs)) then None else
            let arg_constraints = CCList.map2 gen_mk fargs gargs in
                generalize_aux (arg_constraints @ rest) h    
        | _ -> None

    (** [generalizes large small] returns true iff [generalize large small] returns a generalization witness *)
    let generalizes large small = generalize large small
        |> CCOpt.is_some


    (* operational *)

    (* goal: anti-unification, or the least-general generalization *)

    (* join computes the lub *)

    (* join_all computes the lub for a set *)
end


module JSON = struct
    let encode sub = sub
        |> to_assoc
        |> CCList.map (CCPair.map_snd Term.JSON.encode)
        |> JSON.Make.assoc

    let decode json = json
        |> JSON.Parse.(assoc Term.JSON.decode)
        |> CCOpt.map of_assoc
end

module Infix = struct
    let (=?=) = Unification.equate
    let ($) = apply
    let (>->) = compose
end
