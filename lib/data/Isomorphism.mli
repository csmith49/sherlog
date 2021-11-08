(* type def *)

type ('a, 'b) t = {
    to_right : 'a -> 'b;
    to_left : 'b -> 'a;
}

(* construction *)

val iso : ('a -> 'b) -> ('b -> 'a) -> ('a, 'b) t

val id : ('a, 'a) t

val singleton : 'a -> ('a, unit) t

(* combinators *)

val flip : ('a, 'b) t -> ('b, 'a) t

val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

val pair : ('a, 'b) t -> ('c, 'd) t -> ('a * 'c, 'b * 'd) t

(* infix *)

module Infix : sig
    val ( *> ) : 'a -> ('a, 'b) t -> 'b
    val ( <* ) : ('a, 'b) t -> 'b -> 'a

    val ( >> ) : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
end