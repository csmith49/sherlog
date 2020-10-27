module Term : sig
    type t = Watson.Term.t

    val to_json : t -> Yojson.Basic.t
    val of_json : Yojson.Basic.t -> t option
end

module Obligation : sig
    type t = Watson.Obligation.t

    val to_json : t -> Yojson.Basic.t
    val of_json : Yojson.Basic.t -> t option
end

module Atom : sig
    type t = Watson.Atom.t

    val to_json : t -> Yojson.Basic.t
    val of_json : Yojson.Basic.t -> t option

    val conjunct_to_json : t list -> Yojson.Basic.t
    val conjunct_of_json : Yojson.Basic.t -> t list option
end

module Rule : sig
    type t = Watson.Rule.t

    val to_json : t -> Yojson.Basic.t
    val of_json : Yojson.Basic.t -> t option
end

module Program : sig
    type t = Watson.Program.t

    val to_json : t -> Yojson.Basic.t
    val of_json : Yojson.Basic.t -> t option
end