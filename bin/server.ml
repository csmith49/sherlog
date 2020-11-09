(* reference for argument parsing *)
let port = ref 7999 (* default port, changeable via args *)
let verbose = ref false (* whether or not we print out results when handling messages *)

(* argument parsing *)
let spec_list = [
    ("--port", Arg.Set_int port, "Port to host local server on");
    ("--verbose", Arg.Set verbose, "Enable status updates to StdOut");
]
let usage_msg = "Server for SherLog"
let _ = Arg.parse spec_list print_endline usage_msg

let verbose_print str = if !verbose then print_endline str else ()

(* message handling *)

(* for evaluation purposes, we keep some state *)
let program = ref (Watson.Program.of_list [])

(* decompose json into commands/messages/arguments *)
let decompose json =
    let command = Interface.JSON.Parse.(find string   "command"   json) in
    let message = Interface.JSON.Parse.(find identity "message"   json) in
    let args    = Interface.JSON.Parse.(find identity "arguments" json) in
    match command, message, args with
        | Some command, Some message, Some args -> Some (command, message, args)
        | Some command, Some message, None      -> Some (command, message, `Null) (* args are optional *)
        | _                                     -> None

let handler json = match decompose json with
    (* echo the input - for test purposes *)
    | Some ("echo", message, _) -> Some message
    (* parse the string as if it were the contents of a file *)
    | Some ("parse", `String message, _) ->
        let problem = message |> Sherlog.parse in
            Some (Sherlog.Problem.to_json problem)
    (* register a provided program as a piece of global state *)
    | Some ("register", prog, _) -> begin match Sherlog.Program.of_json prog with
        | Some prog ->
            let _ = program := prog in
            Some (`Assoc [("result", `String "success")])
        | _ -> None end
    (* evaluate the provided query on the stored program *)
    | Some ("query", query, _) -> begin match Sherlog.Query.of_json query with
        | Some query ->
            let proof = query
                |> Watson.Proof.of_query
                |> Watson.Proof.resolve !program in
            let model = Sherlog.Model.of_proof proof in
                Some (Sherlog.Model.to_json model)
        | _ -> None end
    | _ -> None

(* main *)
let _ =
    let socket = Interface.Network.socket Interface.Network.local_address !port in
    let server = Interface.Network.server handler socket in
    Interface.Network.run server