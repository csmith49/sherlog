module Problem : sig
    type t

    val to_string : t -> string
    val to_json : t -> Yojson.Basic.t

    module Parameter : sig
        type domain =
            | Unit
            | Positive
            | Real

        val domain_to_string : domain -> string

        type t = Parameter of string * domain

        val to_string : t -> string
        val to_json : t -> Yojson.Basic.t
    end
    
    module Namespace : sig
        type t = Namespace of string
    
        val to_string : t -> string
        val to_json : t -> Yojson.Basic.t
    end

    module Evidence : sig
        type t =
            | Evidence of Watson.Atom.t list
            | ParameterizedEvidence of (string list) * string * Watson.Atom.t list
    
        val to_string : t -> string
        val to_json : t -> Yojson.Basic.t
    end

    val program : t -> Watson.Program.t
    val queries : t -> (Watson.Atom.t list) list
    val parameters : t -> Parameter.t list
    val namespaces : t -> Namespace.t list
    val evidence : t -> Evidence.t list
end

module Program : sig
    type t = Watson.Program.t

    val of_json : Yojson.Basic.t -> t option
end

module Query : sig
    type t = Watson.Atom.t list

    val of_json : Yojson.Basic.t -> t option
end

module Model : sig
    type t

    val to_json : t -> Yojson.Basic.t

    val of_proof : Watson.Proof.t -> t
end

val parse : string -> Problem.t