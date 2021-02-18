(* reference for argument parsing *)
let port = ref 7999 (* default port, changeable via args *)

(* argument parsing *)
let spec_list = [
    ("--port", Arg.Set_int port, "Port to host local server on");
]
let usage_msg = "Server for SherLog"
let _ = Arg.parse spec_list print_endline usage_msg

(* message handling *)

let handler json = match JSON.Parse.(find string "command" json) with
    | Some "parse" ->
        JSON.Parse.(find string "program" json)
            |> CCOpt.map Sherlog.IO.parse
            |> CCOpt.map Sherlog.Program.JSON.encode
    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server handler socket in
    Network.run server