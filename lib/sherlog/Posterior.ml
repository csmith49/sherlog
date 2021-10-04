module Feature = struct
    type t =
        | Size

    let apply proof feature = match feature with
        | Size ->
            let algebra = fun _ -> CCList.fold_left CCInt.( + ) 1 in
            let count = Data.Tree.eval algebra proof in
                CCFloat.of_int count

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

    let apply featurization ensemble = match ensemble with
        | Linear weights -> CCList.map2 CCFloat.( * ) featurization weights
            |> CCList.fold_left CCFloat.( + ) 0.0

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

let featurize proof posterior = posterior.features
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