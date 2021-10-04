module Feature = struct
    type t =
        | Size

    let apply _ feature = match feature with
        | Size -> 0.0

    module JSON = struct
        let encode = function
            | Size -> `Assoc [
                    ("type", `String "feature");
                    ("kind", `String "size");
                ]

        let decode json = let open CCOpt in
            let* kind = JSON.Parse.(find string "kind" json) in
            match kind with
                | "size" -> Some Size
                | _ -> None
    end
end

module Ensemble = struct
    type t =
        | Linear of float list

    let apply embedding ensemble = match ensemble with
        | Linear weights -> CCList.map2 ( *. ) embedding weights
            |> CCList.fold_left ( +. ) 0.0

    module JSON = struct
        let encode = function
            | Linear w -> `Assoc [
                    ("type", `String "ensemble");
                    ("kind", `String "linear");
                    ("weights", w |> CCList.map JSON.Make.float |> JSON.Make.list);
            ]

        let decode json = let open CCOpt in
            let* kind = JSON.Parse.(find string "kind" json) in
            match kind with
                | "linear" -> let* weights = JSON.Parse.(find (list float) "weights" json) in return (Linear weights)
                | _ -> None
    end
end

type t = {
    features : Feature.t list;
    ensemble : Ensemble.t;
}

let embed proof posterior = posterior.features
    |> CCList.map (Feature.apply proof)

let score featurization posterior = posterior.ensemble |> Ensemble.apply featurization

let default = {
    features = [Feature.Size];
    ensemble = Linear [1.0];
}

module JSON = struct
    let encode posterior = `Assoc [
        ("type", `String "posterior");
        ("features", posterior.features |> CCList.map Feature.JSON.encode |> JSON.Make.list);
        ("ensemble", posterior.ensemble |> Ensemble.JSON.encode);
    ]

    let decode json = let open CCOpt in
        let* features = JSON.Parse.(find (list Feature.JSON.decode) "features" json) in
        let* ensemble = JSON.Parse.(find Ensemble.JSON.decode "ensemble" json) in
            return {
                features=features;
                ensemble=ensemble;
            }
end