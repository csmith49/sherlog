(* references for cmd-line parsing *)
let files = ref []
let search_depth = ref CCInt.max_int
let search_width = ref CCInt.max_int
let echo = ref false

let spec_list = [
    ("--depth", Arg.Set_int search_depth, "Sets proof search depth");
    ("--width", Arg.Set_int search_width, "Sets proof search width");
    ("--echo", Arg.Set echo, "If enabled, prints the program before execution");
]
let anon_fun arg = files := arg :: !files
let usage_msg = "Sherlog Interpreter"
let _ = Arg.parse spec_list anon_fun usage_msg

let _ = print_endline ("Found " ^ (!files |> CCList.length |> CCInt.to_string) ^ " file(s).")

(* utility functions *)
let filter = Sherlog.Program.Filter.(
    length !search_depth >> width !search_width >> intro_consistent
)

(* operation to be done per-file *)
let operate filename =
    let _ = print_string ("Processing file " ^ filename ^ "...") in
    (* grab file contents *)
    let contents =
        let channel = open_in filename in
        let length = in_channel_length channel in
        really_input_string channel length in
    let _ = print_string "loaded..." in
    (* convert to program *)
    let program = Sherlog.IO.parse contents in
    let _ = print_string "parsed..." in
    (* pop out the relevant semantics *)
    let facts = program
        |> Sherlog.Program.evidence
        |> CCList.map Sherlog.Evidence.to_fact in
    let _ = print_endline ((facts |> CCList.length |> CCInt.to_string) ^ " facts found.") in
    let _ = if !echo then CCList.iter (fun r -> print_endline (Watson.Rule.to_string r)) (Sherlog.Program.rules program) else () in
    CCList.iter (fun fact ->
        let _ = print_endline ("Fact: " ^ (Watson.Fact.to_string fact)) in
        let _ = print_string   "  Deriving proofs..." in
        let proofs = Sherlog.Program.prove program Sherlog.Program.Filter.total fact in
        let _ = print_endline ("found " ^ (proofs |> CCList.length |> CCInt.to_string) ^ " proofs.") in
        let _ = CCList.iter (fun proof ->
            let _ = print_endline (proof |> Watson.Proof.to_fact |> Watson.Fact.to_string) in
            let contradictions = Sherlog.Program.contradict program filter proof in
            let _ = print_endline ("Found " ^ (contradictions |> CCList.length |> CCInt.to_string) ^ " contradictions.") in
            ()
        ) proofs in
        ()
    ) facts


(* main loop *)
let _ =
    if CCList.length !files > 0 then
        CCList.iter operate !files
    else
        print_endline "No input files provided."