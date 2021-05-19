(* reference for argument parsing *)
let port = ref 7999 (* default port, changeable via args *)
let timeout = ref 3600 (* doesn't do anything atm - lwt requires cooperative threading, and sherlog isn't cooperating *)

(* argument parsing *)
let spec_list = [
    ("--port", Arg.Set_int port, "Port to host local server on");
    ("--timeout", Arg.Set_int timeout, "Max time (in seconds) per query");
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
        let operator = JSON.Parse.(find (list string) "contexts" json)
            |> CCOpt.get_or ~default:[]
            |> Sherlog.Posterior.Feature.context_operator in
        let parameterization = JSON.Parse.(find Sherlog.Posterior.Parameterization.JSON.decode "parameters" json)
            |> CCOpt.get_or ~default:(CCList.replicate (CCList.length operator) 1.0) in
        (* build score function *)
        let score = Sherlog.Posterior.Score.dot parameterization operator in
        (* build filter from parameters *)
        let pos_filter = Sherlog.Program.Filter.(
            intro_consistent
                >> length search_length
                >> beam_width score search_width
        ) in
        let neg_filter = Sherlog.Program.Filter.(
            intro_consistent
                >> length search_length
                >> beam_width score (10 * search_width)
        ) in
        let models = query
            |> Sherlog.Program.models program pos_filter neg_filter
            |> CCList.map Sherlog.Model.JSON.encode in
        return (`List models)
    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server ~timeout:(float_of_int !timeout) handler socket in
    Network.run server