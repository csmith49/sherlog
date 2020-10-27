let parse string = string
    |> Lexing.from_string
    |> Parser.problem Lexer.read
    |> Problem.of_lines

module Problem = Problem
module Model = Model

module Program = Utility.Program

module Query = struct
    type t = Watson.Atom.t list

    let of_json = Utility.Atom.conjunct_of_json
end