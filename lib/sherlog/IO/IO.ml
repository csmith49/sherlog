(* IO Utility *)

(* convert lexing position to string *)
let position_to_string lexbuf =
    let pos = lexbuf.Lexing.lex_curr_p in
    let line = pos.Lexing.pos_lnum in
    let start = pos.Lexing.pos_cnum - pos.Lexing.pos_bol + 1 in
        Printf.sprintf "Line %d, Char. %d" line start

(* Exception for parsing failure *)
exception Sherlog_IO of string

(* Line Stuff *)

let program_of_lines lines =
    (* storage *)
    let rules = ref [] in
    let parameters = ref [] in
    (* handle lines one by one *)
    let process line = match line with
        | `Rule r -> rules := r :: !rules
        | `Parameter p -> parameters := p :: !parameters
        | _ -> () in
    let _ = CCList.iter process lines in
    Program.make !rules !parameters

let evidence_of_lines lines =
    let process line = match line with
        | `Evidence evidence -> Some evidence
        | _ -> None in
    CCList.filter_map process lines

(* Parsing *)

let parse source =
    let lexbuf = Lexing.from_string source in
    try Parser.lines Lexer.read lexbuf with
        | Parser.Error ->
            let message = Fmt.str "Cannot parse %s @ %s" (Lexing.lexeme lexbuf) (position_to_string lexbuf) in
                raise (Sherlog_IO message)
        | Lexer.Error m ->
            let message = Fmt.str "Cannot lex @ %s: %s" (position_to_string lexbuf) m in
                raise (Sherlog_IO message)