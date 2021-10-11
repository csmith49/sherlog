module Embedding = struct
    type t = float list

    let linear weights embedding =
        CCList.map2 ( *. ) weights embedding
            |> CCList.fold_left ( +. ) 0.0

    let stack values = values

    module JSON = struct
        let encode f = f |> JSON.Encode.(list float)
        let decode = JSON.Parse.(list float)
    end
end

module Choice = struct
    type t = {
        embedding : Embedding.t;
        context : Embedding.t list;
    }

    module Functional = struct
        let embedding choice = choice.embedding
        let context choice = choice.context
    end

    module JSON = struct
        let encode choice = `Assoc [
            ("type", `String "choice");
            ("embedding", choice.embedding |> Embedding.JSON.encode);
            ("context", choice.context |> JSON.Encode.list Embedding.JSON.encode);
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

    let choose (values : 'a list) (embed : 'a -> Embedding.t) (score : Embedding.t -> float): ('a * t) CCRandom.t = fun state ->
        let embeddings = CCList.map embed values in
        let scores = CCList.map score embeddings in
        let value, embedding = Random.prop_to (CCList.map2 CCPair.make values embeddings) scores
            |> CCRandom.run ~st:state in
        value, {
            embedding=embedding;
            context=embeddings;
        }
end

module History = struct
    type t = Choice.t list

    let empty = []

    let append choice history = choice :: history

    module JSON = struct
        let encode history = `Assoc [
            ("type", `String "history");
            ("choices", history |> JSON.Encode.list Choice.JSON.encode);
        ]

        let decode json = let open CCOpt in
            let* choices = JSON.Parse.(find "choices" (list Choice.JSON.decode) json) in
            return choices
    end
end

module type Structure = sig
    type candidate

    val stop : candidate -> bool
    val next : candidate -> candidate list

    val embed : candidate -> Embedding.t
    val score : Embedding.t -> float
end

let rec random_walk : type a . (module Structure with type candidate = a) -> a -> (a * History.t) CCRandom.t = fun (module S) -> fun start -> fun state ->
    rw_aux (module S) start History.empty |> CCRandom.run ~st:state
and rw_aux : type a . (module Structure with type candidate = a) -> a -> History.t -> (a * History.t) CCRandom.t = fun (module S) -> fun value -> fun history -> fun state ->
    (* check if we've met the early stopping condition *)
    if S.stop value then (value, history)
    else match S.next value with
        (* no follow-ups *)
        | [] -> (value, history)
        | candidates ->
            let value, choice = Choice.choose candidates S.embed S.score
                |> CCRandom.run ~st:state in
            let history = History.append choice history in
                rw_aux (module S) value history state