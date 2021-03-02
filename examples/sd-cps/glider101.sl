# PARAMETERS ---------------------

# material and component selection
!parameter hull_material : Discrete["steel", "aluminum", ...].
!parameter battery : Discrete["A", "AA", "AAA", "C", "D", ...].
!parameter fin : Discrete["fin_001", "fin_002", ...].
!parameter wings : Discrete["wing_001", "wing_002", ...].

# the real- and natural-valued parameters below have bogus domain ranges

# hull shape
!parameter hull_inner_diameter : Real[0, 100].
!parameter hull_outer_diameter : Real[0, 100].
!parameter hull_length : Real[0, 100].

# energy configuration
!parameter batteries_per_series : Natural[0, 100].
!parameter series_per_layer : Natural[0, 100].
!parameter layers_per_design : Natural[0, 100].

# PARAMETER RELATIONS ------------

# hull dimensions

hull_id(hull_inner_diameter).
hull_od(hull_outer_diameter).
hull_thickness(T) <- hull_id(I), hull_od(O), T = (O - I) / 2.
hull(hull_material).

# physical components

fin(fin).
wings(wings).

# batteries are packed into a configuration (B, S, G, L):
# 1. battery of type B
# 2. groups of S batteries in series
# 3. G groups per layer
# 4. L layers

packing_configuration(battery, batteries_per_series, series_per_layer, layers_per_design).

# ENERGY CONSTRAINTS -------------

packing_voltage(B, S, V) <- voltage(B, V'), V = S * V'.
packing_energy(B, S, G, L, E) <- watt_hours(B, Wh), E = Wh * S * G * L.

# rely on external function to determine packing dimensions

packing_diameter(B, S, G; packomania[D, N]) <- battery_diameter(B, D), N = S * G.
packing_length(B, L, Length) <- battery_length(B, L'), Length = L * L'.

# make sure we produce enough voltage and energy to satisfy mission recs

!constrain
    packing_configuration(B, S, G, L),
    packing_voltage(B, S, V), voltage_requirements(V_req), V >= V_req,
    packing_energy(B, S, G, L, E), energy_requirements(E_req), E >= E_req.

# HULL GEOMETRY

# first computation only holds for thin walls
hoop_stress(P, I, O, S) <-
    T < I / 20,                             # determination for thin walls
    T = (O - I) / 2,                        # thickness from ID and OD
    S = P * I / T.                          # stress computation

# second, for thick walls
hoop_stress(P, I, O, S) <-
    T >= I / 20,                            # determination for thick walls
    T = (O - I) / 2,                        # thickness from ID and OD,
    R_I = I / 2, R_O = O / 2,               # compute radii
    S = -2 * P * R_O^2 / (R_O^2 - R_I^2).   # stress computation

# the above are disjoint, but there's probably a cleaner way to represent that

# make sure the hull is thick enough to withstand the required pressure

!constrain
    hull_id(I), hull_od(O), pressure_requirements(P),
    hull(M), material_yield(M, Y), safety_factor(F),
    hoop_stress(P, I, O, S), S <= Y / F.

# DRAG

# all the below relies on the following external function
# drag[G, M, S, D], where:
# G - geometry of the relevant part
# M - part material, used for friction coefficients
# S - speed of travel
# D - depth of travel, used to determine kinematic viscosity of seawater

# drag[] is either a CFD simulation, or some approximation thereof
# already a first-order approximation ignoring interference drag

wing_drag(W, S, D; drag[G, M, S, D]) <- wing_geometry(W, G), wing_material(W, M).
fin_drag(F, S, D; drag[G, M, S, D]) <- fin_geometry(F, G), fin_material(F, M).
body_drag(O, L, M, S, D; drag[G, M, S, D]) <- hull_geometry(O, L, G).

total_drag(O, L, M, W, F, S, D, T) <-
    wing_drag(W, S, D, Wd),
    fin_drag(F, S, D, Fd),
    body_drag(O, L, M, S, D, Bd),
    T = Wd + Fd + Bd.

# drag determines how much lift we need for a given speed
# believe this computation is non-linear? cast it to external computation for now

sufficient_lift(D, S; lift[D, S]).

# we probably want to be able to write statements like the above
# not currently allowed with our semantics - D and S are not guarded in the body

# make sure our wing generates enough thrust to get up to max speed

!constrain
    hull_od(O), hull_length(L), hull_material(M), wing(W), fin(F),
    speed_requirements(S), depth_requirements(D),
    total_drag(O, L, M, W, F, S, D, T),
    sufficient_lift(T, S, Lift), wing_lift(W, Lift'), Lift' >= Lift.