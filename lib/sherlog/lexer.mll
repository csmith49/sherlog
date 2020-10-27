{
    open Parser

    exception ParseError of string
}

(* whitespace *)
let space = ' '
let tab = '\t'
let newline = '\n'

(* alphas *)
let lower = ['a'-'z']
let upper = ['A'-'Z']

(* symbols *)
let prime = '''
let underscore = '_'
let period = '.'
let comment_start = '#'

(* numbers *)
let digit = ['0'-'9']
let neg = '-'

(* lexing rules *)
rule read = parse
    (* whitespace *)
    | (space | tab )+ { read lexbuf }
    | newline { Lexing.new_line lexbuf; read lexbuf }
    (* comment *)
    | comment_start [^'\n']* { read lexbuf }
    (* brackets *)
    | "(" { LPARENS }
    | ")" { RPARENS }
    | "[" { LBRACKET }
    | "]" { RBRACKET }
    (* symbols *)
    | ";" { SEMICOLON }
    | "<-" | ":-" { ARROW }
    | ":-" { ARROW } (* not the default, but we should at least support tradition *)
    | ":" { COLON }
    | "@" { AT }
    | "." { PERIOD }
    | "," { COMMA }
    (* inference tokens *)
    | "!parameter" { PARAMETER }
    | "!namespace" { NAMESPACE }
    | "!evidence" { EVIDENCE }
    (* queries *)
    | "?" { QMARK }
    (* eof *)
    | eof { EOF }
    (* boolean *)
    | "true" { TRUE }
    | "false" { FALSE }
    (* numbers *)
    | neg? digit+ period digit+ { FLOAT (float_of_string (Lexing.lexeme lexbuf)) }
    | neg? digit+ { INTEGER (int_of_string (Lexing.lexeme lexbuf)) }
    (* symbols and variables *)
    | upper (lower | upper | digit | underscore)* prime* { VARIABLE (Lexing.lexeme lexbuf)}
    | lower (lower | upper | digit | underscore)* { SYMBOL (Lexing.lexeme lexbuf) }
    (* escape case *)
    | _ { raise (ParseError ("Found " ^ (Lexing.lexeme lexbuf) ^ ": don't know how to handle")) }