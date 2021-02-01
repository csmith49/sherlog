type t = Yojson.Basic.t
type 'a parser = Yojson.Basic.t -> 'a option

module Parse = struct
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

    let find parser key json = match json with
        | `Assoc xs -> xs
            |> CCList.assoc_opt ~eq:CCString.equal key
            |> CCOpt.flat_map parser
        | _ -> None

    let identity = CCOpt.return

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
end

module Make = struct
    let string s = `String s
    let int i = `Int i
    let float f = `Float f
    let bool b = `Bool b
    let list xs = `List xs
    let assoc xs = `Assoc xs
    let null = `Null
end