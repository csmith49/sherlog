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

(* utility *)
let rec shuffle xs = match xs with
    | [] -> []
    | xs ->
        (* get a random index *)
        let idx' = xs
            |> CCList.length
            |> CCRandom.int in
        let idx = CCRandom.run idx' in
        (* split xs into x and xs' *)
        let x = CCList.get_at_idx_exn idx xs in
        let xs' = CCList.remove_at_idx idx xs in
            x :: (shuffle xs')

(* message handling *)

let handler json = let open CCOpt in match JSON.Parse.(find "type" string json) with
    | Some "parse-source-request" ->
        let* source = JSON.Parse.(find "source" string json) in
        let lines = source |> Sherlog.IO.parse in
        let response = `Assoc [
            ("type", `String "parse-source-response");
            ("program", lines |> Sherlog.IO.program_of_lines |> Sherlog.Program.JSON.encode);
            ("evidence", lines
                |> Sherlog.IO.evidence_of_lines
                |> JSON.Encode.list Sherlog.Evidence.JSON.encode
            );
        ] in
        return response

    | Some "query-request" ->
        (* core programmatic info *)
        let* program = JSON.Parse.(find "program" Sherlog.Program.JSON.decode json) in
        let* query = JSON.Parse.(find "evidence" Sherlog.Evidence.JSON.decode json) in
        (* get explanation *)
        let explanation = query
            |> Sherlog.Evidence.to_atoms
            |> fun cs -> Sherlog.Program.proof cs program
            |> fun (proof, history) -> Sherlog.Explanation.of_proof proof history
            |> Sherlog.Explanation.JSON.encode in
        let response = `Assoc [
            ("type", `String "query-response");
            ("explanation", explanation)
        ] in
        return response

    | _ -> None

(* main *)
let _ =
    let socket = Network.socket Network.local_address !port in
    let server = Network.server ~timeout:(float_of_int !timeout) handler socket in
    Network.run server