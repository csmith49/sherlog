type ('a, 'b) t = {
    embed : 'a -> 'b;
    pullback : 'b -> 'a option;
}

let embed embedding = embedding.embed
let pullback embedding = embedding.pullback

let mk embed pullback = {
    embed = embed;
    pullback = pullback;
}

let compose left right = {
    embed = (fun a -> a |> left.embed |> right.embed);
    pullback = (fun c -> c |> right.pullback |> CCOpt.flat_map left.pullback);
}

let id = {
    embed = (fun a -> a);
    pullback = (fun a -> Some a);
}

let pair left right = {
    embed = (fun (a, c) -> (embed left a, embed right c));
    pullback = (fun (b, d) ->
        CCOpt.map2 CCPair.make (pullback left b) (pullback right d)
    );
}

let list embedding = {
    embed = CCList.map (embed embedding);
    pullback = (fun xs -> xs
        |> CCList.map (pullback embedding)
        |> CCList.all_some);
}