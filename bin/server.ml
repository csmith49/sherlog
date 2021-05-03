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
    | Some "query" -> let open CCOpt in
        (* get core program and query *)
        let* program = JSON.Parse.(find Sherlog.Program.JSON.decode "program" json) in
        let* query = JSON.Parse.(find Sherlog.Evidence.JSON.decode "query" json) 
            |> CCOpt.map Sherlog.Evidence.to_atoms in
        (* get parameters for search *)
        let search_length = JSON.Parse.(find int "depth" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        let search_width = JSON.Parse.(find int "width" json)
            |> CCOpt.get_or ~default:CCInt.max_int in
        (* build filter from parameters *)
        let filter = Sherlog.Program.Filter.(
            intro_consistent >> length search_length >> width search_width
        ) in
        let models = query
            |> Sherlog.Program.models program filter
            |> CCList.map Sherlog.Model.JSON.encode in
        return (`List models)
    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server handler socket in
    Network.run server