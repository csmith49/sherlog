(* Trees are directed acyclic graphs with nodes annotated by a label *)
type 'a tree = Node of 'a * 'a tree list

(* Even though the constructor is exposed, we can access the label and children functionally *)
val label : 'a tree -> 'a
val children : 'a tree -> 'a tree list

(* Leaves are nodes with no out-edges *)
val leaf : 'a -> 'a tree
val is_leaf : 'a tree -> bool

(* Paths are ordered lists of labels *)
type 'a path = 'a list
val paths : 'a tree -> 'a path list
val of_path : 'a path -> 'a tree option

(* Tree zippers *)
type 'a zipper
val zipper : 'a tree -> 'a zipper
val of_zipper : 'a zipper -> 'a tree

(* Basic zipper movement *)
val up : 'a zipper -> 'a zipper option
val down : 'a zipper -> 'a zipper option
val left : 'a zipper -> 'a zipper option
val right : 'a zipper -> 'a zipper option

(* fancier zipper movement *)
val next : 'a zipper -> 'a zipper option
val preorder : 'a zipper -> 'a zipper option

val find : ('a tree -> bool) -> 'a zipper -> 'a zipper option
val find_all : ('a tree -> bool) -> 'a zipper -> 'a zipper list

(* Extracting info from zippers *)
val focus : 'a zipper -> 'a tree
val set_focus : 'a tree -> 'a zipper -> 'a zipper
val path_to_focus : 'a zipper -> 'a path

(* evaluation and collapses *)
type ('a, 'b) algebra = 'a -> 'b list -> 'b
val eval : ('a, 'b) algebra -> 'a tree -> 'b