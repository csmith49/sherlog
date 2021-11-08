type t = Choice.t list

let choices history = history

let append choice history = choice :: history

let empty = []


let score history = history
    |> choices
    |> CCList.map Choice.score
    |> CCList.fold_left ( *. ) 1.0

module JSON = struct
    let encode history = `Assoc [
        ("type", `String "history");
        ("choices", JSON.Encode.list Choice.JSON.encode history);
    ]

    let decode json = let open CCOpt in
        let* choices = JSON.Parse.(find "choices" (list Choice.JSON.decode) json) in
        return choices
end