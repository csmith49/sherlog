(* utility functions *)
let gumbel_trick weight = let open CCRandom in
	let scale u = u |> log |> log |> fun x -> 0.0 -. x in
	(float_range 0.0 1.0) >|= scale >|= CCFloat.add (log weight)

let categorical weights : int CCRandom.t = fun state ->
	let scores = weights
    	|> CCList.map gumbel_trick
    	|> CCList.map (CCRandom.run ~st:state)
    	|> CCList.mapi CCPair.make in
  	let sort_key (li, ls) (ri, rs) = if ls >= rs then (li, ls) else (ri, rs) in
  	let argmax = scores |> CCList.fold_left sort_key (0, -1.0) in
  		fst argmax

let prop_to (values : 'a list) (score : 'a -> float) : 'a CCRandom.t = fun state ->
	let scores = CCList.map score values in
	let index = CCRandom.run ~st:state (categorical scores) in
		values |> CCList.get_at_idx index |> CCOpt.get_exn_or "Invalid sampling index"

(* signature for linear search structure *)
module type Domain = sig
	type t
	val features : t -> float list
	val score : float list -> float
	val accept : t -> bool
	val reject : t -> bool
	val successors : t -> t list
end

module History = struct
	module Record = struct
		type t = float list * float list list

		let make features context = (features, context)

		let features = fst

		module JSON = struct
			let encode_float_list fs = fs
				|> CCList.map JSON.Make.float
				|> JSON.Make.list

			let encode record = `Assoc [
				("type", `String "record");
				("features", record |> fst |> encode_float_list);
				("context", record |> snd |> CCList.map encode_float_list |> JSON.Make.list);
			]
			let decode json = let open CCOpt in
				let* features = JSON.Parse.(find (list float) "features" json) in
				let* context = JSON.Parse.(find (list (list float)) "context" json) in
					return (features, context)
		end
	end
	
	type t = Record.t list

	let extend history record = record :: history
	let empty = []

	let most_recent_features history = CCList.hd history |> Record.features

	module JSON = struct
		let encode history = `Assoc [
			("type", `String "history");
			("records", history |> CCList.map Record.JSON.encode |> JSON.Make.list);
		]

		let decode json = let open CCOpt in
			let* records = JSON.Parse.(find (list Record.JSON.decode) "records" json) in
				return records
	end
end

module State = struct
	type 'a t = 'a * History.t
	let make value history = (value, history)
	let value = fst
	let history = snd
	let to_tuple state = (value state, history state)
	let check pred state = state |> fst |> pred
	let extend_history state record = CCPair.map_snd (fun h -> History.extend h record) state
	let features state = state |> history |> History.most_recent_features
end

let rec stochastic_beam_search : type a . (module Domain with type t = a) -> int -> a State.t list -> a State.t list -> a State.t list = fun (module D) width worklist accepted -> match worklist with
	(* case 1: empty worklist, just return all found solutions *)
	| [] -> accepted
	(* case 2: worklist populated, so check, expand, and recurse *)
	| states ->
		(* divide states into rejected / accepted / in-progress *)
		let newly_accepted, in_progress = states
			|> CCList.partition (State.check D.reject) |> snd
			|> CCList.partition (State.check D.accept) in
		(* expand the worklist with D's successor function *)
		(* step 1: unpack the successors of each state *)
		let unpack state =
			let succs = state |> State.value |> D.successors in
			let history = state |> State.history in
			let build value = (value, D.features value, history) in
				CCList.map build succs in
		let unpacked = in_progress |> CCList.flat_map unpack in
		(* step 2: extract the full context to rebuild histories *)
		let features (_, fs, _) = fs in
		let context = CCList.map features unpacked in
		(* step 3: repack to states *)
		let repack (v, fs, h) =
			let record = History.Record.make fs context in
			let history = History.extend h record in
				State.make v history in
		let repacked = unpacked |> CCList.map repack in
		(* step 4: sub-sample, if necessary *)
		let worklist = 
			if CCList.length repacked <= width then repacked 
			else
				let scores state = state |> State.features |> D.score in
				let samples = prop_to repacked scores
					|> CCList.replicate width
					|> CCRandom.list_seq in
				CCRandom.run samples in
		(* recurse *)
		stochastic_beam_search (module D) width worklist (newly_accepted @ accepted)