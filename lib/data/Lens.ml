(* LENS *)

type ('a, 'b) t = {
    get : 'a -> 'b;
    set : 'b -> 'a -> 'a;
}

let get lens = lens.get
let set lens = lens.set

let mk get set = {
    get = get;
    set = set;
}

(* UTILITY *)

(* not exposed, but common *)
let ( $ ) f g x = g (f x)

let modify lens f = fun a ->
    lens.set (f @@ lens.get a) a

let compose left right = {
    get = left.get $ right.get;
    set = (fun c a -> modify left (right.set c) a);
}

(* CONSTRUCTION *)

let null = {
    get = (fun _ -> ());
    set = (fun _ a -> a);
}

let id = {
    get = (fun a -> a);
    set = (fun b _ -> b);
}

(* TUPLES *)

let pair left right = {
    get = (fun (a, c) -> (left.get a, right.get c));
    set = (fun (a, c) (b, d) -> (left.set a b, right.set c d));
}

let fst = {
    get = fst;
    set = (fun b (_, a) -> (b, a) );
}

let snd = {
    get = snd;
    set = (fun b (a, _) -> (a, b) );
}

(* LISTS *)

let list lens = {
    get = CCList.map lens.get;
    set = CCList.map2 lens.set;
}

let hd = {
    get = CCList.hd;
    set = (fun b a -> b :: CCList.tl a);
}

let tl = {
    get = CCList.tl;
    set = (fun b a -> (CCList.hd a) :: b)
}

(* ASSOCS *)

let assoc key = {
    get = CCList.Assoc.get_exn ~eq:CCString.equal key;
    set = CCList.Assoc.set ~eq:CCString.equal key;
}

(* INFIX *)

module Infix = struct
    let ( @ ) obj lens = lens.get obj

    let ( <@ ) lens value = lens.set value

    let ( >> ) left right = compose left right
end