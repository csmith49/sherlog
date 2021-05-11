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
        (* build score function *)
        let pos_score = Sherlog.Posterior.(score_of_assoc [
            (* (0.5, Feature.constrained_intros); *)
            (* (0.3, Feature.intros); *)
            (* (-1.0, Feature.length); *)
            (0.2, Feature.context "fuzzy:stress");
            (0.1, Feature.context "fuzzy:asthma_spontaneous");
            (0.3, Feature.context "fuzzy:asthma_comorbid");
            (0.3, Feature.context "fuzzy:influence");
        ]) in
        (* build filter from parameters *)
        let pos_filter = Sherlog.Program.Filter.(
            intro_consistent
                >> length search_length
                >> beam_width pos_score search_width
        ) in
        let neg_score = Sherlog.Posterior.(score_of_assoc [
            (0.8, Feature.context "fuzzy:stress");
            (0.9, Feature.context "fuzzy:asthma_spontaneous");
            (0.7, Feature.context "fuzzy:asthma_comorbid");
            (0.7, Feature.context "fuzzy:influence");
        ]) in
        let neg_filter = Sherlog.Program.Filter.(
            intro_consistent
                >> length search_length
                >> beam_width neg_score (10 * search_width)
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