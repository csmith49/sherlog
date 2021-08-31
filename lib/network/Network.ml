open Lwt

(* exposed type aliases *)
type handler = Yojson.Basic.t -> Yojson.Basic.t option
type port = int
type address = Lwt_unix.inet_addr
type socket = Lwt_unix.file_descr
type server = unit -> unit Lwt.t

(* the local address *)
let local_address = Unix.inet_addr_loopback

(* max timeout info *)
let max_timeout = 3600.0

(* constructs a connection from a socket *)
(* let connection_of_socket socket =
    let ic = Lwt_io.of_fd ~mode:Lwt_io.Input socket in
    let oc = Lwt_io.of_fd ~mode:Lwt_io.Output socket in
        (ic, oc)

*)
let safe_parse string = try Some (Yojson.Basic.from_string string) with _ -> None
(*
let safe_handle handler json = try handler json with _ -> None

(* applies a handler to a connection *)
let rec handle_connection handler (ic, oc) () = Lwt_io.read_line_opt ic >>= fun msg ->
    (* attempt to parse and handle the message *)
    let result = msg
        |> CCOpt.flat_map safe_parse
        |> CCOpt.flat_map (safe_handle handler) in
    let reply = match result with
        | Some result -> Yojson.Basic.to_string result
        | None -> Yojson.Basic.to_string (`String "failure") in
    Lwt_io.write_line oc reply >>= handle_connection handler (ic, oc)

let exn_response connection exn =
    let _, oc = connection in
    let response = match exn with
        | Lwt_unix.Timeout -> `String "timeout"
        | _ -> `String "unknown" in
    Lwt_io.write_line oc (response |> Yojson.Basic.to_string) *)

(* constructs a handler for the output of lwt_unix.accept *)
(* let handle_socket time handler (socket, _) =
    let _ = time in
    let connection = connection_of_socket socket in
    let promise () = Lwt_unix.with_timeout 10.0 (handle_connection handler connection) in
    catch promise (exn_response connection) *)

(* constructs a socket *)
let socket address port = let open Lwt_unix in
    let sock = socket PF_INET SOCK_STREAM 0 in
    let addr = ADDR_INET (address, port) in
    let _ = bind sock addr in
    let _ = listen sock 10 in
    let _ = setsockopt sock SO_REUSEPORT true in (* ensures we can restart the server quickly *)
        sock

let read connection =
    let fd = fst connection in
    let ic = Lwt_io.of_fd ~mode:Lwt_io.Input fd in
        Lwt_io.read_line_opt ic >|= CCOpt.flat_map safe_parse

let write connection json =
    let fd = fst connection in
    let oc = Lwt_io.of_fd ~mode:Lwt_io.Output fd in
    let message = match json with
        | Some json -> Yojson.Basic.to_string json
        | None -> let msg = `Assoc [
            ("type", `String "failure");
            ("message", `String "Handler failed to produce response.");
        ] in
        Yojson.Basic.to_string msg in
    Lwt_io.write_line oc message
(* 
let handle_with_timeout (timeout : float) (handler : Yojson.Basic.t -> Yojson.Basic.t option) (message : Yojson.Basic.t option) : Yojson.Basic.t option t= match message with
    | None -> return None
    | Some json ->
        let response_promise () = wrap1 handler json in
        let timeout_promise _ = return None in
        catch (fun () -> Lwt_unix.with_timeout timeout response_promise) timeout_promise *)


let handle_with_timeout (timeout : float) (handler : Yojson.Basic.t -> Yojson.Basic.t option) (message : Yojson.Basic.t option) : Yojson.Basic.t option t= match message with
| None -> return None
| Some json ->
    let _ = timeout in
    let timeout_promise = Lwt_unix.sleep 0.01 >>= fun () -> return_none in
    let response_promise = wrap1 handler json in
    pick [timeout_promise; response_promise]

let accept_connection timeout handler connection =
    let rec respond () =
        read connection >>= handle_with_timeout timeout handler >>= write connection >>= respond
    in respond ()

let server ?timeout:(time=max_timeout) message_handler socket =
    let rec serve () = Lwt_unix.accept socket >>= accept_connection time message_handler >>= serve
    in serve

let run server = Lwt_main.run (server ())