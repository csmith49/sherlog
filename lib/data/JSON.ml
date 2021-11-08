type t = Yojson.Basic.t

type 'a embedding = ('a, t) Embedding.t

(* builtins *)

let string =
    let embed s = `String s in
    let pullback = function
        | `String s -> Some s
        | _ -> None in
    Embedding.mk embed pullback

let int =
    let embed i = `Int i in
    let pullback = function
        | `Int i -> Some i
        | _ -> None in
    Embedding.mk embed pullback

let float =
    let embed f = `Float f in
    let pullback = function
        | `Float f -> Some f
        | _ -> None in
    Embedding.mk embed pullback

let bool =
    let embed b = `Bool b in
    let pullback = function
        | `Bool b -> Some b
        | _ -> None in
    Embedding.mk embed pullback

let null =
    let embed () = `Null in
    let pullback = function
        | `Null -> Some ()
        | _ -> None in
    Embedding.mk embed pullback

let list embedding =
    let embedding = Embedding.list embedding in
    let embed xs = `List (Embedding.embed embedding xs) in
    let pullback = function
        | `List xs -> Embedding.pullback embedding xs
        | _ -> None in
    Embedding.mk embed pullback

let assoc embedding =
    let embedding = embedding
        |> Embedding.pair Embedding.id
        |> Embedding.list in
    let embed xs = `Assoc (Embedding.embed embedding xs) in
    let pullback = function
        | `Assoc xs -> Embedding.pullback embedding xs
        | _ -> None in
    Embedding.mk embed pullback

(* utility *)
let find key embedding json = json
    |> Embedding.pullback (assoc embedding)
    |> CCOpt.flat_map (CCList.Assoc.get ~eq:CCString.equal key)