let program_of_lines lines =
    (* storage *)
    let rules = ref [] in
    let dependencies = ref [] in
    let constraints = ref [] in
    let parameters = ref [] in
    let evidence = ref [] in
    (* handle lines one by one *)
    let process line = match line with
        | `Rule       r ->        rules := r :: !rules
        | `Dependency d -> dependencies := d :: !dependencies
        | `Constraint c ->  constraints := c :: !constraints
        | `Parameter  p ->   parameters := p :: !parameters
        | `Evidence   e ->     evidence := e :: !evidence in
    let _ = CCList.iter process lines in
    (* then build *)
    let ontology = Ontology.make !dependencies !constraints in
    Program.make !rules !parameters !evidence ontology

let position_to_string lexbuf =
    let pos = lexbuf.Lexing.lex_curr_p in
    let line = pos.Lexing.pos_lnum in
    let start = pos.Lexing.pos_cnum - pos.Lexing.pos_bol + 1 in
        Printf.sprintf "Line %d, Char. %d" line start
    
    let parse string =
    let lexbuf = Lexing.from_string string in
    let lines = try Parser.lines Lexer.read lexbuf with
        | Parser.Error ->
            let _ = Printf.fprintf stderr "Parse Error @ %s" (position_to_string lexbuf) in
            exit (-1)
        | Lexer.Error m ->
            let _ = Printf.fprintf stderr "Lexer Error @ %s: %s" (position_to_string lexbuf) m in
            exit (-1)
    in program_of_lines lines
    