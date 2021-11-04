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

module Random = struct
    let gumbel_trick w = let open CCRandom in
    let scale u = u |> log |> log |> fun x -> 0.0 -. x in
        (float_range 0.0 1.0) >|= scale >|= CCFloat.add (log w)

    let categorical weights = fun state ->
        let scores = weights
            |> CCList.map gumbel_trick
            |> CCList.map (CCRandom.run ~st:state)
            |> CCList.mapi CCPair.make in
        let sort_key (li, ls) (ri, rs) = if ls >= rs then (li, ls) else (ri, rs) in
        let argmax = scores |> CCList.fold_left sort_key (0, -1.0) in
            fst argmax

    let prop_to values scores = fun state ->
        let index = CCRandom.run ~st:state (categorical scores) in
        values |> CCList.get_at_idx index |> CCOpt.get_exn_or "Invalid sampling."
end

let choose (values : 'a list) (embed : 'a -> Embedding.t) : ('a * t) CCRandom.t = fun state ->
    let embeddings = CCList.map embed values in
    let scores = CCList.map Embedding.score embeddings in
    let value, embedding = Random.prop_to (CCList.map2 CCPair.make values embeddings) scores
        |> CCRandom.run ~st:state in
    value, make embedding embeddings