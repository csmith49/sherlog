(* linear combination of features *)
type t = {
    features : float list;
    weights : float list;
}

let features embedding = embedding.features
let weights embedding = embedding.weights

let score embedding = CCList.map2 ( *. ) embedding.features embedding.weights
    |> CCList.fold_left ( +. ) 0.0

let of_features pairs candidate =
    let weights = CCList.map fst pairs in
    let features = CCList.map (fun (_, f) -> f candidate) pairs in
    {
        features = features;
        weights = weights;
    }

module JSON = struct
    let encode embedding = `Assoc [
        ("type", `String "embedding");
        ("features", JSON.Encode.(list float) embedding.features);
        ("weights", JSON.Encode.(list float) embedding.weights);
    ]

    let decode json = let open CCOpt in
        let* features = JSON.Parse.(find "features" (list float) json) in
        let* weights = JSON.Parse.(find "weights" (list float) json) in
        return {
            features = features;
            weights = weights;
        }
end