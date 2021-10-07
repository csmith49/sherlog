type 'a t = 'a Statement.t list

module Functional = struct
    let statements pl = pl
    
    let make stmts = stmts
end

module JSON = struct
    let encode encoder pl = `Assoc [
        ("type", JSON.Encode.string "pipeline");
        ("statements", pl |> Functional.statements |> JSON.Encode.(list (Statement.JSON.encode encoder)));
    ]
end