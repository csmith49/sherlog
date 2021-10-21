type ('a, 'b) t

val embed : ('a, 'b) t -> 'a -> 'b
val pullback : ('a, 'b) t -> 'b -> 'a option

val mk : ('a -> 'b) -> ('b -> 'a option) -> ('a, 'b) t

val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

(** {2 Construction} *)

val id : ('a, 'a) t

val pair : ('a, 'b) t -> ('c, 'd) t -> ('a * 'c, 'b * 'd) t

val list : ('a, 'b) t -> ('a list, 'b list) t