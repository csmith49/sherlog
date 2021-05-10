(* references for cmd-line parsing *)
let files = ref []
let search_depth = ref CCInt.max_int
let search_width = ref CCInt.max_int
let echo = ref false
let color = ref false

let spec_list = [
    ("--depth", Arg.Set_int search_depth, "Sets proof search depth");
    ("--width", Arg.Set_int search_width, "Sets proof search width");
    ("--echo", Arg.Set echo, "If enabled, prints the program before execution");
    ("--color", Arg.Set color, "Enables color output");
]
let anon_fun arg = files := arg :: !files
let usage_msg = "Starting Sherlog interpreter..."
let _ = Arg.parse spec_list anon_fun usage_msg

(* for printing *)
let marker = (Fmt.styled (`Faint) (Fmt.const Fmt.string "[🔍]"))
let _ = if !color then Fmt.set_style_renderer Fmt.stdout `Ansi_tty else ()

(* header *)
let _ = Fmt.pr "%a %s\n" marker () usage_msg
let _ = Fmt.pr "%a Search width/depth: %a/%a\n" 
    marker ()
    (Fmt.styled (`Fg `Blue) Fmt.int) !search_width
    (Fmt.styled (`Fg `Blue) Fmt.int) !search_depth

(* utility functions *)
let filter = Sherlog.Program.Filter.(
    intro_consistent >> length !search_depth >> uniform_width !search_width
)

(* operation to be done per-file *)
let operate filename =
    let _ = Fmt.pr "%a Processing file %a...\n" 
        marker () 
        (Fmt.styled (`Fg `Blue) Fmt.string) filename in
    
    (* grab file contents *)
    let contents =
        let channel = open_in filename in
        let length = in_channel_length channel in
        really_input_string channel length in
    
    (* convert to program *)
    let program = Sherlog.IO.parse contents in
    
    (* pop out the relevant semantics *)
    let facts = program
        |> Sherlog.Program.evidence
        |> CCList.map Sherlog.Evidence.to_atoms in
    let _ = Fmt.pr "%a Found %a fact(s).\n" 
        marker ()
        (Fmt.styled (`Fg `Blue) Fmt.int) (facts |> CCList.length) in
    
    (* print the program if echo is on *)
    (* TODO - Program.pp needs cleanup *)
    let _ = if !echo
        then Fmt.pr "%a Converted program: %a\n" marker () Sherlog.Program.pp program
        else () in

    (* processing a proof *)
    let process_model index model =
        let _ = Fmt.pr "%a Examining model %a...\n"
            marker ()
            (Fmt.styled (`Fg `Blue) Fmt.int) index in
        let _ = Fmt.pr "%a Model: %a\n"
            marker ()
            Sherlog.Model.pp model in
        () in
    
    (* processing a fact *)
    let process_fact fact =
        (* render the found fact *)
        let _ = Fmt.pr "%a Fact: @[%a@]\n"
            marker ()
            (Fmt.list ~sep:Fmt.comma Watson.Atom.pp) fact in
        (* compute proofs and process *)
        let models = Sherlog.Program.models program filter fact in
        let _ = Fmt.pr "%a Found %a models.\n"
            marker ()
            (Fmt.styled (`Fg `Blue) Fmt.int) (models |> CCList.length) in
        let _ = CCList.iteri process_model models in
        () in

    CCList.iter process_fact facts

(* main loop *)
let _ =
    if CCList.length !files > 0 then
        let _ = Fmt.pr "%a Found %a file(s).\n" 
            marker () 
            (Fmt.styled (`Fg `Blue) Fmt.int) (!files |> CCList.length) in
        CCList.iter operate !files
    else
        Fmt.pr "%a No input files provided.\n" marker ()