type t = Yojson.Basic.t

type 'a embedding = ('a, t) Embedding.t

(* builtins *)

val string : string embedding
val int : int embedding
val float : float embedding
val bool : bool embedding
val null : unit embedding

val list : 'a embedding -> 'a list embedding
val assoc : 'a embedding -> (string * 'a) list embedding

(* utility *)
val find : string -> 'a embedding -> t -> 'a option