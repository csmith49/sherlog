module Feature : sig
    type t

    val apply : Proof.t -> t -> float

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

module Ensemble : sig
    type t

    val apply : Search.Featurization.t -> t -> float

    module JSON : sig
        val encode : t -> JSON.t
        val decode : JSON.t -> t option
    end
end

type t

val featurize : Proof.t -> t -> Search.Featurization.t
val score : Search.Featurization.t -> t -> float

module JSON : sig
    val encode : t -> JSON.t
    val decode : JSON.t -> t option
end