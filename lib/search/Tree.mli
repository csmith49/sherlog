(* search trees *)
type ('a, 'b) t =
    | Interior of ('a, 'b) edge list
    | Exterior of status * 'b
and ('a, 'b) edge = Edge of 'a * ('a, 'b) t
and status =
    | Frontier
    | Success
    | Failure

val state : ('a, 'b) t -> 'b option

(* simple constructors *)
val success : 'b -> ('a, 'b) t
val failure : 'b -> ('a, 'b) t
val frontier : 'b -> ('a, 'b) t
val interior : ('a * 'b) list -> ('a, 'b) t

(* check if the tree is a frontier in the search *)
val is_frontier : ('a, 'b) t -> bool

(* search paths *)
type ('a, 'b) path = Path of 'a list * 'b

(* extracting successful paths *)
val paths_to_success : ('a, 'b) t -> ('a, 'b) path list
val witnesses_along_path : ('a, 'b) path -> 'a list

module Zipper : sig
    (* zippers for manipulation of trees *)
    type ('a, 'b) zipper

    (* getters and setters *)
    val focus : ('a, 'b) zipper -> ('a, 'b) t
    val set_focus : ('a, 'b) t -> ('a, 'b) zipper -> ('a, 'b) zipper

    (* basic movement *)
    val left : ('a, 'b) zipper -> ('a, 'b) zipper option
    val right : ('a, 'b) zipper -> ('a, 'b) zipper option
    val up : ('a, 'b) zipper -> ('a, 'b) zipper option
    val down : ('a, 'b) zipper -> ('a, 'b) zipper option

    (* ~advanced~ movement *)
    val next : ('a, 'b) zipper -> ('a, 'b) zipper option
    val preorder : ('a, 'b) zipper -> ('a, 'b) zipper option

    val find : (('a, 'b) t -> bool) -> ('a, 'b) zipper -> ('a, 'b) zipper option

    (* conversions *)
    val of_tree : ('a, 'b) t -> ('a, 'b) zipper
    val to_tree : ('a, 'b) zipper -> ('a, 'b) t
end
