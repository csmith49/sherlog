let program_of_lines lines =
    (* storage *)
    let rules = ref [] in
    let parameters = ref [] in
    let evidence = ref [] in
    (* handle lines one by one *)
    let process line = match line with
        | `Rule       r ->        rules := r :: !rules
        | `Parameter  p ->   parameters := p :: !parameters
        | `Evidence   e ->     evidence := e :: !evidence in
    let _ = CCList.iter process lines in
    (* then build *)
    Program.make !rules !parameters !evidence 

let position_to_string lexbuf =
    let pos = lexbuf.Lexing.lex_curr_p in
    let line = pos.Lexing.pos_lnum in
    let start = pos.Lexing.pos_cnum - pos.Lexing.pos_bol + 1 in
        Printf.sprintf "Line %d, Char. %d" line start

exception Sherlog_IO of string

let parse st =
    let lexbuf = Lexing.from_string st in
    let lines = try Parser.lines Lexer.read lexbuf with
        | Parser.Error ->
            let message = Fmt.str "Cannot parse %s @ %s" (Lexing.lexeme lexbuf) (position_to_string lexbuf) in
                raise (Sherlog_IO message)
        | Lexer.Error m ->
            let message = Fmt.str "Cannot lex @ %s: %s" (position_to_string lexbuf) m in
                raise (Sherlog_IO message)
    in program_of_lines lines