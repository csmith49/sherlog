(* Utility module for producing and consuming Yojson objects. *)

type t = Yojson.Basic.t

type 'a parser = t -> 'a option
type 'a encoder = 'a -> t

module Parse = struct
    (* basic types *)
    
    let string = function
        | `String s -> Some s
        | _ -> None

    let int = function
        | `Int i -> Some i
        | _ -> None

    let float = function
        | `Float f -> Some f
        | _ -> None

    let bool = function
        | `Bool b -> Some b
        | _ -> None

    let null = function
        | `Null -> Some ()
        | _ -> None

    (* combinators *)

    let list parser json = match json with
        | `List xs -> xs
            |> CCList.map parser
            |> CCList.all_some
        | _ -> None

    let assoc parser json = match json with
        | `Assoc xs -> xs
            |> CCList.map (fun (k, v) -> match parser v with
                | Some v -> Some (k, v)
                | _ -> None)
            |> CCList.all_some
        | _ -> None

    (* utility *)

    let find key parser json = match json with
        | `Assoc xs -> xs
            |> CCList.assoc_opt ~eq:CCString.equal key
            |> CCOpt.flat_map parser
        | _ -> None
end

module Encode = struct
    (* basic types *)

    let string s = `String s

    let int i = `Int i

    let float f = `Float f

    let bool b = `Bool b

    let null () = `Null

    (* combinators *)

    let list encoder ls = `List (CCList.map encoder ls)

    let assoc encoder ls = `Assoc (CCList.map (CCPair.map_snd encoder) ls)
end