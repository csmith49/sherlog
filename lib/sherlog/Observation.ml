module Literal = struct
    type t =
        | Equal of string * Model.Value.t
        | NotEqual of string * Model.Value.t

    let variable = function
        | Equal (variable, _) -> variable
        | NotEqual (variable, _) -> variable

    let value = function
        | Equal (_, value) -> value
        | NotEqual (_, value) -> value

    let eq variable value = Equal (variable, value)
    let neq variable value = NotEqual (variable, value)

    let pp ppf = function
        | Equal (variable, value) -> Fmt.pf ppf "%s == %a" variable Model.Value.pp value
        | NotEqual (variable, value) -> Fmt.pf ppf "%s != %a" variable Model.Value.pp value

    module JSON = struct
        let encode = function
            | Equal (variable, value) -> `Assoc [
                ("type", `String "equal");
                ("variable", `String variable);
                ("value", Model.Value.JSON.encode value)
            ]
            | NotEqual (variable, value) -> `Assoc [
                ("type", `String "not equal");
                ("variable", `String variable);
                ("value", Model.Value.JSON.encode value)
            ]
    end
end

type t = Literal.t list

let eq_of_assoc xs = xs |> CCList.map (CCPair.merge Literal.eq)

let pp ppf observation = Fmt.pf ppf "%a" (Fmt.list ~sep:Fmt.comma Literal.pp) observation

module JSON = struct
    let encode observation = `Assoc [
        ("type", `String "observation");
        ("literals", (JSON.Encode.list Literal.JSON.encode) observation);
    ]
end