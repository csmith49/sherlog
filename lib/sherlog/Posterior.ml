module Feature = struct
    type t =
        | Size

    let apply feature branch = match feature with
        | Size -> branch
            |> Branch.witnesses
            |> CCList.length
            |> CCFloat.of_int

    module JSON = struct
        let encode = function
            | Size -> `Assoc [
                    ("type", `String "features");
                    ("kind", `String "size");
                ]

        let decode json = let open CCOpt in
            let* kind = JSON.Parse.(find "kindd" string json) in
            match kind with
                | "size" -> Some Size
                | _ -> None
    end
end

type t = (float * Feature.t) list


let embedding posterior = posterior
    |> CCList.map (CCPair.map_snd Feature.apply)
    |> Search.Embedding.of_features

let apply posterior branch = embedding posterior branch

module JSON = struct
    let encode posterior = `Assoc [
        ("type", `String "posterior");
        ("features", posterior.features |> JSON.Encode.list Feature.JSON.encode);
        ("ensemble", posterior.ensemble |> Ensemble.JSON.encode);
    ]

    let decode json = let open CCOpt in
        let* features = JSON.Parse.(find "features" (list Feature.JSON.decode) json) in
        let* ensemble = JSON.Parse.(find "ensemble" Ensemble.JSON.decode json) in
            return {
                features=features;
                ensemble=ensemble;
            }
end