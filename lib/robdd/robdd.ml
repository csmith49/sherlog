(* variables *)
type variable = string
type order = variable list

(* nodes are either terminal (0, 1) or an indexed decision *)
type node =
    | Terminal of bool
    | Decision of int

let rec node_hash node = CCInt.hash (int_of_node node)
and int_of_node = function
    | Terminal false -> 0
    | Terminal true -> 1
    | Decision n -> n + 2

let node_compare left right = (int_of_node right) - (int_of_node left)

module IntMap = CCMap.Make(CCInt)

type robdd = {
    labels : variable IntMap.t;
    ordering : variable -> int;
    pointers : (node * node) IntMap.t;
    cache : int IntMap.t;
}

let variable robdd = function
    | Terminal _ -> None
    | Decision decision -> IntMap.find_opt decision robdd.labels

let low robdd = function
    | Terminal _ -> None
    | Decision decision -> IntMap.find_opt decision robdd.pointers |> CCOpt.map fst

let high robdd = function
    | Terminal _ -> None
    | Decision decision -> IntMap.find_opt decision robdd.pointers |> CCOpt.map snd

let hash variable low high = (variable, (low, high))
    |> CCHash.pair CCString.hash (CCHash.pair node_hash node_hash)

let node robdd variable low high =
    (* check if it already exists in the cache *)
    match IntMap.find_opt (hash variable low high) robdd.cache with
        | Some decision -> Decision decision, robdd
        (* if not, make it and add it *)
        | None ->
            let decision = IntMap.cardinal robdd.pointers in
            let robdd = {
                labels = IntMap.add decision variable robdd.labels;
                ordering = robdd.ordering;
                pointers = IntMap.add decision (low, high) robdd.pointers;
                cache = IntMap.add (hash variable low high) decision robdd.cache;
            } in
            Decision decision, robdd

module PointerMap = CCMap.Make(struct
    type t = node * node
    let compare = CCPair.compare node_compare node_compare
end)
