type t = {
    embedding : Embedding.t;
    context : Embedding.t list;
}

let embedding choice = choice.embedding
let context choice = choice.context

let make embedding context = {
    embedding = embedding;
    context = context;
}

let score choice =
    let numerator = Embedding.score choice.embedding in
    let denominator = choice.context
        |> CCList.map Embedding.score
        |> CCList.fold_left ( +. ) 0.0 in
    numerator /. denominator

module JSON = struct
    let encode choice = `Assoc [
        ("type", `String "choice");
        ("embedding", Embedding.JSON.encode choice.embedding);
        ("context", JSON.Encode.list Embedding.JSON.encode choice.context);
    ]

    let decode json = let open CCOpt in
        let* embedding = JSON.Parse.(find "embedding" Embedding.JSON.decode json) in
        let* context = JSON.Parse.(find "context" (list Embedding.JSON.decode) json) in
        return {
            embedding = embedding;
            context = context;
        }
end

let choose (values : 'a list) (embed : 'a -> Embedding.t) : ('a * t) CCRandom.t = fun state ->
    let embeddings = CCList.map embed values in
    let scores = CCList.map Embedding.score embeddings in
    let value, embedding = Utility.prop_to (CCList.map2 CCPair.make values embeddings) scores
        |> CCRandom.run ~st:state in
    value, make embedding embeddings

let choose_k (samples : int) (values : 'a list) (embed : 'a -> Embedding.t) : ('a * t) list CCRandom.t = fun state ->
    let embeddings = CCList.map embed values in
    let scores = CCList.map Embedding.score embeddings in
    let indices = CCRandom.sample_without_duplicates ~cmp:CCInt.compare samples (Utility.categorical scores) state in
    let sample index =
        let value = CCList.get_at_idx_exn index values in
        let embedding = CCList.get_at_idx_exn index embeddings in
        let choice = make embedding embeddings in
        value, choice in
    CCList.map sample indices