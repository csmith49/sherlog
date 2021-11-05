module Operation = struct
    type t =
        | Size

    let apply operation branch = match operation with
        | Size -> branch
            |> Branch.witnesses
            |> CCList.length
            |> CCFloat.of_int

    module JSON = struct
        let encode = function
            | Size -> `Assoc [
                    ("type", `String "operation");
                    ("kind", `String "size");
                ]

        let decode json = let open CCOpt in
            let* kind = JSON.Parse.(find "kind" string json) in
            match kind with
                | "size" -> Some Size
                | _ -> None
    end
end

module Feature = struct
    type t = {
        weight : float;
        operation : Operation.t;
    }

    let apply feature branch = Operation.apply feature.operation branch
    let weight feature = feature.weight

    let tuple feature = (weight feature, apply feature)

    module JSON = struct
        let encode feature = `Assoc [
            ("type", `String "feature");
            ("weight", `Float feature.weight);
            ("operation", Operation.JSON.encode feature.operation);
        ]

        let decode json = let open CCOpt in
            let* weight = JSON.Parse.(find "weight" float json) in
            let* operation = JSON.Parse.(find "operation" Operation.JSON.decode json) in
            return {
                weight = weight;
                operation = operation;
            }
    end
end

type t = Feature.t list

let embedding posterior = posterior
    |> CCList.map Feature.tuple
    |> Search.Embedding.of_features

let apply posterior branch = embedding posterior branch

let of_operations assoc = assoc
    |> CCList.map (fun (w, op) -> {Feature.weight = w; operation = op})

let default = of_operations [(1.0, Operation.Size)]


module JSON = struct
    let encode posterior = `Assoc [
        ("type", `String "posterior");
        ("features", JSON.Encode.list Feature.JSON.encode posterior);
    ]

    let decode json = JSON.Parse.(find "features" (list Feature.JSON.decode) json)
end