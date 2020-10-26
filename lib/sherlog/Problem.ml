module Parameter = struct
    type domain =
        | Unit
        | Positive
        | Real
        | Categorical of int

    let domain_to_string = function
        | Unit -> "[0, 1]"
        | Positive -> "ℝ⁺"
        | Real -> "ℝ"
        | Categorical n -> "Cat<" ^ (string_of_int n) ^ ">"

    type t = Parameter of string * domain

    let to_string = function Parameter (name, domain) -> name ^ " : " ^ (domain_to_string domain)
end

type t = {
    parameters : Parameter.t list;
    program : Watson.Semantics.Program.t;
    observations : Watson.Language.Atom.t list;
}