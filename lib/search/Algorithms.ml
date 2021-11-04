(* SEARCH SIGNATURES *)

module type Space = sig
    type candidate

    val compare : candidate -> candidate -> int
    val is_solution : candidate -> bool
    val successors : candidate -> candidate list
    val embed : candidate -> Embedding.t
end

type 'a search = ('a * History.t) CCRandom.t

let run search = CCRandom.run search

(* ALGORITHMS *)

module BeamSearch (S : Space) = struct
    (* wraps S.candidates with histories *)
    module Element = struct
        type t = {
            candidate : S.candidate;
            history : History.t;
        }

        let of_candidate candidate = {
            candidate = candidate;
            history = History.empty;
        }

        let compare left right = S.compare left.candidate right.candidate
        let equal left right = compare left right == 0

        let is_solution elt = S.is_solution elt.candidate

        let successors elt = elt.candidate
            |> S.successors
            |> CCList.map (fun s -> {candidate = s; history = elt.history;})

        let embed elt = S.embed elt.candidate

        let append_choice choice elt = { elt with history = History.append choice elt.history }

        let tuple elt = (elt.candidate, elt.history)
    end

    (* a list of elements - the current state of the search *)
    module Beam = struct
        type t = Element.t list

        let size = CCList.length

        let expand (beam : t) : t = beam
            |> CCList.flat_map Element.successors 
            |> CCList.uniq ~eq:Element.equal

        let choose beam =
            let repair_elt (elt, choice) = Element.append_choice choice elt in
            Choice.choose beam Element.embed
                |> CCRandom.map repair_elt

        let prune (width : int) (beam : t) : t CCRandom.t =
            (* make sure the beam is wide enough to warrant pruning *)
            if size beam <= width then CCRandom.return beam else
            (* if so, pick [width] samples w/o replacement *)
            CCRandom.sample_without_duplicates ~cmp:Element.compare width (choose beam)

        let solution beam = CCList.find_opt Element.is_solution beam
    end

    let rec search width candidate : S.candidate search =
        search_aux width [Element.of_candidate candidate] |> CCRandom.map Element.tuple
    and search_aux (width : int) (beam : Beam.t) : Element.t CCRandom.t = match Beam.solution beam with
        (* if we can find a solution, just return that *)
        | Some elt -> CCRandom.return elt
        (* otherwise, expand the beam and keep trying *)
        | None -> beam
            |> Beam.expand
            |> Beam.prune width
            |> CCRandom.flat_map (search_aux width)
end

let beam_search : type a . (module Space with type candidate = a) -> int -> a -> a search = fun (module S) -> fun width -> fun candidate ->
    let module Search = BeamSearch(S) in Search.search width candidate