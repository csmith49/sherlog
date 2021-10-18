module Value = struct
    module Unwrapped = struct
        type t =
            | Integer of int
            | Float of float
            | Boolean of bool
            | Function of string * t list
            | Unit

        let rec of_term = function
            | Watson.Term.Integer i -> Some (Integer i)
            | Watson.Term.Float f -> Some (Float f)
            | Watson.Term.Boolean b -> Some (Boolean b)
            | Watson.Term.Function (f, args) -> args
                |> CCList.map of_term
                |> CCList.all_some
                |> CCOpt.map (fun args -> Function (f, args))
            | Watson.Term.Unit -> Some Unit
            | _ -> None

        let rec pp ppf = function
            | Integer i -> Fmt.pf ppf "%d" i
            | Float f -> Fmt.pf ppf "%f" f
            | Boolean true -> Fmt.pf ppf "T"
            | Boolean false -> Fmt.pf ppf "F"
            | Unit -> Fmt.pf ppf "*"
            | Function (f, args) -> Fmt.pf ppf "%s(%a)"
                f
                (Fmt.list ~sep:Fmt.comma pp) args

        let rec equal left right = match left, right with
            | Integer l, Integer r -> CCInt.equal l r
            | Float l, Float r -> CCFloat.equal l r
            | Boolean l, Boolean r -> CCBool.equal l r
            | Unit, Unit -> true
            | Function (f, fargs), Function (g, gargs) ->
                (CCString.equal f g) &&
                (CCList.equal equal fargs gargs)
            | _ -> false

        module JSON = struct
            let rec encode = function
                | Integer i -> `Int i
                | Float f -> `Float f
                | Boolean b -> `Bool b
                | Unit -> `Null
                | Function (f, args) -> `Assoc [
                    ("type", `String "function");
                    ("function_id", `String f);
                    ("arguments", `List (CCList.map encode args));
                ]
        end
    end

    type t = Unwrapped.t Pipe.Value.t

    let of_term = function
        | Watson.Term.Variable id | Watson.Term.Symbol id -> Some (Pipe.Value.Identifier id)
        | (_ as term) -> term
            |> Unwrapped.of_term
            |> CCOpt.map (fun term -> Pipe.Value.Literal term)

    let pp = Pipe.Value.pp Unwrapped.pp

    module JSON = struct
        let encode = Pipe.Value.JSON.encode Unwrapped.JSON.encode
    end
end

module Statement = struct
    type t = Value.Unwrapped.t Pipe.Statement.t

    let of_introduction introduction =
        let target = Introduction.sample_site introduction in
        let function_id = Introduction.Functional.function_id introduction in
        let arguments = Introduction.Functional.arguments introduction 
            |> CCList.map Value.of_term
            |> CCList.all_some in
        arguments |> CCOpt.map (Pipe.Statement.Functional.make target function_id)

    let make target function_id terms =
        let arguments = terms
            |> CCList.map Value.of_term
            |> CCList.all_some in
        arguments
            |> CCOpt.map (Pipe.Statement.Functional.make target function_id)

    let pp = Pipe.Statement.pp Value.Unwrapped.pp

    let equal = Pipe.Statement.equal Value.Unwrapped.equal

    module JSON = struct
        let encode = Pipe.Statement.JSON.encode Value.Unwrapped.JSON.encode
    end
end

type t = Value.Unwrapped.t Pipe.Pipeline.t

let of_statements = Pipe.Pipeline.Functional.make

let pp = Pipe.Pipeline.pp Value.Unwrapped.pp

module JSON = struct
    let encode = Pipe.Pipeline.JSON.encode Value.Unwrapped.JSON.encode
end