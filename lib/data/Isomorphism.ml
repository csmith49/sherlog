(* type def *)

type ('a, 'b) t = {
    to_right : 'a -> 'b;
    to_left : 'b -> 'a;
}

let ( $ ) f g x = g (f x)

(* construction *)

let iso to_right to_left = {
    to_right = to_right;
    to_left = to_left;
}

let id = {
    to_right = (fun x -> x);
    to_left = (fun x -> x);
}

let singleton value = {
    to_right = (fun _ -> ());
    to_left = (fun () -> value);
}

(* combinators *)

let flip iso = {
    to_right = iso.to_left;
    to_left = iso.to_right;
}

let compose f g = {
    to_right = f.to_right $ g.to_right;
    to_left = g.to_left $ f.to_left;
}

let pair fst snd = {
    to_right = (fun (a, c) -> (fst.to_right a, snd.to_right c));
    to_left = (fun (b, d) -> (fst.to_left b, snd.to_left d))
}

(* infix *)

module Infix = struct
    let ( *> ) a iso = iso.to_right a
    let ( <* ) iso b = iso.to_left b

    let ( >> ) f g = compose f g
end