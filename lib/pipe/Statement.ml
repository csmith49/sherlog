type 'a t = {
    target : string;
    function_id : string;
    arguments : 'a Value.t list;
}

module Functional = struct
    let target stmt = stmt.target
    let function_id stmt = stmt.function_id
    let arguments stmt = stmt.arguments

    let make target function_id arguments = {
        target=target;
        function_id=function_id;
        arguments=arguments;
    }
end

module JSON = struct
    let encode encoder stmt = `Assoc [
        ("type", JSON.Encode.string "statement");
        ("target", stmt |> Functional.target |> JSON.Encode.string);
        ("function_id", stmt |> Functional.function_id |> JSON.Encode.string);
        ("arguments", stmt |> Functional.arguments |> JSON.Encode.(list (Value.JSON.encode encoder)));
    ]
end