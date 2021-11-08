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