type 'a t = 'a Statement.t list

let pp lit_pp ppf pl = Fmt.pf ppf "%a"
    (Fmt.list (Statement.pp lit_pp)) pl

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