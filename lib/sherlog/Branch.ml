type t = {
    witnesses : Watson.Proof.Witness.t list;
    status : status;
}
and status =
    | Frontier of Watson.Proof.Obligation.t
    | Terminal of bool

let status_compare left right = match left, right with
    | Frontier left, Frontier right -> Watson.Proof.Obligation.compare left right
    | Terminal left, Terminal right -> CCBool.compare left right
    | Frontier _, _ -> -1
    | _, Frontier _ -> 1

let witness_compare = CCList.compare Watson.Proof.Witness.compare

let compare left right =
    let witness_compare = witness_compare left.witnesses right.witnesses in
    if witness_compare != 0 then witness_compare else
        status_compare left.status right.status
let equal left right = (compare left right) == 0

(* getters *)

let witnesses branch = branch.witnesses
let status branch = branch.status

let obligation branch = match status branch with
    | Frontier obligation -> Some obligation
    | _ -> None
let success branch = match status branch with
    | Terminal success -> Some success
    | _ -> None

(* setters *)

let set_success success branch = { branch with status = Terminal success }
let set_frontier obligation branch = { branch with status = Frontier obligation }
let add_witness witness branch = { branch with witnesses = witness :: branch.witnesses }

(* construction *)

let of_conjunct atoms = 
    let obligation = Watson.Proof.Obligation.of_conjunct atoms in {
        witnesses = [];
        status = Frontier obligation;
    }

(* rule application *)

let resolve branch rule = let open CCOpt in
    let* obligation = obligation branch in
    let* (witness, obligation) = Watson.Proof.resolve obligation rule in
    branch
        |> add_witness witness
        |> set_frontier obligation
        |> return
    
let is_resolved branch = branch
    |> obligation
    |> CCOpt.map Watson.Proof.Obligation.is_empty
    |> CCOpt.get_or ~default:false

(* manipulation *)

let extend rules branch = 
    (* case 1 - obligation unresolved *)
    if is_resolved branch then [branch |> set_success true]
    (* case 2 - try to apply some rules! *)
    else match CCList.filter_map (resolve branch) rules with
        (* case 2.a - no valid rule applications *)
        | [] -> [branch |> set_success false]
        (* case 2.b - some rules apply, just return *)
        | successors -> successors

(* Evaluation *)

module type Algebra = sig
    type result

    val terminal : bool -> result
    val frontier : Watson.Proof.Obligation.t -> result
    val witness : result -> Watson.Proof.Witness.t -> result
end

let eval : type a . (module Algebra with type result = a) -> t -> a = fun (module A) -> fun branch ->
    let result = match status branch with
        | Terminal success -> A.terminal success
        | Frontier obligation -> A.frontier obligation in
    branch
        |> witnesses
        |> CCList.fold_left A.witness result