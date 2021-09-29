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
    type uni_c = Uni of Term.t * Term.t
    let uni_mk l r = Uni (l, r)
    let uni_map f = function Uni (l, r) -> Uni (f l, f r)

    let rec unify left right = unify_aux [ Uni (left, right) ] empty
    (* simple implementation of Martelli-Montanari unification *)
    and unify_aux eqs h = match eqs with
        | [] -> Some h
        | Uni (x, y) :: rest when Term.equal x y -> unify_aux rest h
        | Uni (Term.Variable x, y) :: rest | Uni (y, Term.Variable x) :: rest  -> 
            if Term.occurs x y then None else
            let h = compose h (singleton x y) in
            let rest = CCList.map (uni_map (apply h)) rest in
                unify_aux rest h
        | Uni (Term.Function (f, fargs), Term.Function (g, gargs)) :: rest ->
            if (CCString.equal f g) && (CCInt.equal (CCList.length fargs) (CCList.length gargs)) then None else
            let sub_eqs = CCList.map2 uni_mk fargs gargs in
                unify_aux (sub_eqs @ rest) h
        | _ -> None
end

module Generalization = struct
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

    (** [join l r] returns a term [lgg] such that [lgg] generalizes [l] and [r] and specializes any other generalization *)
    let rec join left right = match left, right with
        | x, y when Term.equal x y -> x
        | Term.Function (f, fargs), Term.Function (g, gargs) when CCString.equal f g && CCInt.equal (CCList.length fargs) (CCList.length gargs) ->
            Term.Function (f, CCList.map2 join fargs gargs)
        | x, y ->
            let tag = CCHash.(pair Term.hash Term.hash) (x, y) in
            let name = "lgg" in
            Term.Make.Variable.tagged name tag
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
    let ($) = apply
    let (>->) = compose
end
