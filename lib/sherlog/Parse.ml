let parse_string string = string
    |> Lexing.from_string
    |> Parser.problem Lexer.read