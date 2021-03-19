# BATTERY ATTRIBUTES (Examples) ----------------------------------------------

# All easily read from spec sheets.
diameter('D', 34.2). # mm
length('D', 61.5).   # mm
volume('D', 5190).   # mm^3
mass('D', 170).    # g
voltage('D', 2.5).   # V
capacity('D', 40).   # Wh

# Not always given in such a direct form - `reliability`% to have capacity +/- `margin`.
reliability('D', 0.95).
margin('D', 0.2).

# PACKING CONFIGURATION ------------------------------------------------------

!parameter battery            : Discrete['D', 'A', 'AA', 'AAA'].
!parameter batteries_per_cell : Natural.
!parameter cells_per_layer    : Natural.
!parameter layers_per_config  : Natural.

# Configurations represented by tuple (B, C, L), where:
# B - batteries per cell (wired in series)
# C - cells per layer
# L - layers per configuration

number_of_batteries((B, C, L), N) <- N = B * C * L.

# INDUCED BATTERY AND PACKING ATTRIBUTES -------------------------------------

# We compute actual capacity by using reliability information for those in-spec.
# Note we treat each battery independently - represent by the pair (T, I), where:
# T - type of battery (D, AA, etc.)
# I - index of battery

in_spec((T, I); Bernoulli[R]) <- int(I), reliability(T, R).

actual_capacity((T, I); Uniform[C - M, C + M]) <-
    in_spec((T, I), true),
    capacity(T, C), margin(T, M).

# Batteries out-of-spec need some assumption on distribution. Options include:
# 1. Normal distribution - out-of-spec batteries not likely to be far out of spec.
# 2. Demonic - assume out-of-spec batteries are totally dead.
# 3. Half-Normal distribution - out-of-spec batteries can't have *more* capacity.
# 4. Mixture?
# For this document, go with option 3.

actual_capacity((T, I); Half-Normal['LT', C - M, 0.1]) <-
    in_spec((T, I), false),
    capacity(T, C), margin(T, M).

# Packing voltage determined by batteries per-cell.
packing_voltage(T, (B, _, _), R) <- R = V * B, voltage(T, V).

# Packing capacity depends on capacity of *each* battery.
packing_capacity(T, (B, C, L), R) <-
    pc_aux(T, N, R), number_of_batteries((B, C, L), N).

# Recursively sum capacity of all batteries.
pc_aux(T, 0, 0.0).
pc_aux(T, I, R) <-
    I >= 0, pc_aux(T, I - 1, R'),
    actual_capacity((T, I), C),
    R = C + R'.
    
# Packing diameter given by external table look-up.
packing_diameter(T, (B, C, _); packomania[D, B * C]) <- diameter(T, D).

packing_length(T, (_, _, L), R) <- length(T, L'), R = L * L'.

# Packing volume computable given a configuration C.
packing_volume(T, C, R) <-
    packing_diameter(T, C, D),
    packing_length(T, C, L),
    R = pi * L * (D / 2) ^ 2.

# Packing bouyancy depends on packing volume, battery mass, and seawater density.
packing_bouyancy(T, (B, C, L), R) <-
    packing_volume(T, (B, C, L), V),
    number_of_batteries((B, C, L), N), mass(T, M),
    seawater_density(D),
    R = (M * N) / (V * D).

# DRAG APPROXIMATIONS --------------------------------------------------------

# Assume the rest of the design is parametrized by the battery packing configuration,
# and that we have some mechanism for approximating drag (see, e.g., auv-geometry.sl).
# Drag impacts the capacity requirements for the battery - the more drag, the less
# efficient propulsion is, and the more capacity needed to meet mission requirements
# on range and depth.

drag(T, C; drag_external[D, L]) <- packing_diameter(T, C, D), packing_length(T, C, L).

capacity_requirements(T, C; hotel_and_prop_reqs[D, B]) <- drag(T, C, D), packing_bouyancy(T, C, B).

# BATTERY CONSTRAINTS --------------------------------------------------------

# We want to meet our capacity needs, but which packing configuration maximizes the
# likelihood of that happening? This program encodes an optimization task: find the 
# parameter assignments that maximize the likelihood of the constraint below being
# satisfied.

!evidence
    C = (batteries_per_cell, cells_per_layer, layers_per_config),
    T = battery, safety_factor(F),
    capacity_requirements(T, C, R), packing_capacity(T, C, A),
    A >= (R * F).