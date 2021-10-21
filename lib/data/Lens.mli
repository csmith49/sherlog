(** {1 Lens} *)

(** Functional implementation of lenses. *)
type ('a, 'b) t

val get : ('a, 'b) t -> 'a -> 'b
val set : ('a, 'b) t -> 'b -> 'a -> 'a

val mk : ('a -> 'b) -> ('b -> 'a -> 'a) -> ('a, 'b) t

(** {2 Utility} *)

(** [modify lens f] applies [f] to what the lens is observing *)
val modify : ('a, 'b) t -> ('b -> 'b) -> 'a -> 'a

(** [compose left right] chains [left] then [right] *)
val compose : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t

(** {2 Construction} *)

val null : ('a, unit) t
val id : ('a, 'a) t

(** {3 Tuples} *)

val pair : ('a, 'b) t -> ('c, 'd) t -> ('a * 'c, 'b * 'd) t

val fst : ('a * 'b, 'a) t
val snd : ('a * 'b, 'b) t

(** {3 Lists} *)

val list : ('a, 'b) t -> ('a list, 'b list) t

val hd : ('a list, 'a) t
val tl : ('a list, 'a list) t

(** {3 Assocs} *)

val assoc : string -> ((string * 'a) list, 'a) t

(** {2 Infix Operations} *)

module Infix : sig
    (* alias for get *)
    val ( @ ) : 'a -> ('a, 'b) t -> 'b

    (* alias for set *)
    val ( <@ ) : ('a, 'b) t -> 'b -> 'a -> 'a

    (* alias for compose *)
    val ( >> ) : ('a, 'b) t -> ('b, 'c) t -> ('a, 'c) t
end