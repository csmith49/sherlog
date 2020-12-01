let position_to_string lexbuf =
    let pos = lexbuf.Lexing.lex_curr_p in
    let line = pos.Lexing.pos_lnum in
    let start = pos.Lexing.pos_cnum - pos.Lexing.pos_bol + 1 in
        Printf.sprintf "Line %d, Char. %d" line start

let parse string =
    let lexbuf = Lexing.from_string string in
    let problem = try Parser.problem Lexer.read lexbuf with
        | Parser.Error ->
            let _ = Printf.fprintf stderr "Parse Error @ %s" (position_to_string lexbuf) in
            exit (-1)
        | Lexer.Error m ->
            let _ = Printf.fprintf stderr "Lexer Error @ %s: %s" (position_to_string lexbuf) m in
            exit (-1)
    in Problem.of_lines problem

module Problem = Problem
module Model = Model

module Program = Utility.Program

module Query = struct
    type t = Watson.Atom.t list

    let of_json = Utility.Atom.conjunct_of_json
end