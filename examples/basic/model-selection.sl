# Parameters
!parameter prior : unit.
!parameter control : unit.
!parameter treated : unit.
!parameter uniform : unit.

# Facts
control(p_0).
control(p_1).
control(p_2).
control(p_3).
control(p_4).

treated(p_5).
treated(p_6).
treated(p_7).
treated(p_8).
treated(p_9).

# Rules
effectiveness(true; bernoulli[prior]).
effective <- effectiveness(_, 1.0).
ineffective <- effectiveness(_, 0.0).

outcome(P; bernoulli[uniform]) <- ineffective, person(P).
outcome(P; bernoulli[control]) <- effective, control(P).
outcome(P; bernoulli[treated]) <- effective, treated(P).

person(P) <- treated(P).
person(P) <- control(P).

# Evidence
control_outcome <-
    outcome(p_0, 0.0),
    outcome(p_1, 0.0),
    outcome(p_2, 1.0),
    outcome(p_3, 0.0),
    outcome(p_4, 0.0).

treated_outcome <-
    outcome(p_5, 1.0),
    outcome(p_6, 0.0),
    outcome(p_7, 1.0),
    outcome(p_8, 1.0),
    outcome(p_9, 1.0).

!evidence control_outcome, treated_outcome.