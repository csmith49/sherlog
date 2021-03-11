# AUV Sizing Spreadsheet Conversion

# DRIVING VARIABLES -> PARAMETERS

# geometry (in meters)
!parameter diameter : positive.
!parameter length : positive.

# hull material, simplified
!parameter material : discrete('aluminum', 'titanium', 'stainless', ...).

# depth to which vessel will travel (in meters)
!parameter depth_rating : positive.

# safety factor for unmanned vehicles (unitless)
!parameter safety_factor : positive.

# batteries to power vehicle (Wh/kg)
!parameter battery_specific_energy : positive.

# fraction of pressure vessel bouyancy used for batteries
!parameter battery_bouyancy_fraction : unit.

# power draw by all non-propulsion systems (Wh/kg)
!parameter hotel_power_draw : positive.

# unitless drag coefficient for computing drag force
!parameter drag_coefficient : real.

# hack to account for control fin drag
!parameter appendage_added_area : unit.

# efficiency of prop and motor, including loss for gearboxes, shaft seals, etc.
!parameter propulsion_efficiency : unit.

# averages out at 1,027 (kg/m^3)
!parameter density_of_seawater : positive.

# REFERENCE GEOMETRY FOR PV
# only transcribing the relevant values
# from the MBARI Tethys LRAUV Design

# (m)
ref_diameter(0.3048).

# (m)
ref_nose_length(0.422).
ref_tail_length(0.889).

# (m^2)
ref_wetted_nose_area(0.4041).
ref_wetted_tail_area(0.5854).

# (m^3)
ref_nose_volume(0.0271).
ref_tail_volume(0.0349).

# GEOMETRY CALCULATIONS FOR PV

total_wetted_area(N + P + T) <- wetted_nose_area(N), wetted_pv_area(P), wetted_tail_area(T).
fairing_displacement(N + P + T) <- nose_volume(N), pv_volume(P), tail_volume(T).
pv_displacement(V * D) <-
    pv_volume(P), density_of_seawater(D),
    R = diameter / 2,
    V = P + (4/3) * pi * (R ^ 3).

diameter_ratio(R) <- ref_diameter(D), R = diameter / D.

nose_length(N * R) <- ref_nose_length(N), diameter_ratio(R).
tail_length(T * R) <- ref_tail_length, diameter_ratio(R).
pv_length(L) <- nose_length(N), tail_length(T), L = length - N - T.

wetted_nose_area(N * R') <- ref_wetted_nose_area(N), diameter_ratio(R), R' = R^2.
wetted_pv_area(pi * V) <- pv_length(P), V = diameter * P.
wetted_tail_area(T * R') <- ref_wetted_tail_area(T), diameter_ratio(R), R' = R^2.

nose_volume(N * R') <- ref_nose_volume(N), diameter_ratio(R), R' = R^3.
pv_volume(A * L) <- L = length, A = pi * (diameter / 2)^2.
tail_volume(T * R') <- ref_tail_volume(T), diameter_ratio(R), R' = R^3.

# PRESSURE VESSEL CALCULATIONS

# assumes each material above has relations for:
# 1. Young's Modulus,
# 2. Yield Stress,
# 3. Poisson's Ratio, and
# 4. Density.

crush_depth(D * F) <- D = depth_rating, F = safety_factor.
crush_pressure(P) <- crush_depth(D), P = D * density_of_seawater * 9.806.

pv_cylinder_thickness_elastic_failure(F) <-
    youngs_modulus(material, Y), poissons_ratio(material, R), crush_depth(D),
    F = diameter * ((D / 2 / Y) * (1 - R))^(1/3).

pv_cylinder_thickness_yield_failure(F) <-
    yield_stress(material, S), crush_depth(D),
    F = diameter * 0.5 * (1 - sqrt(1 - (2 * D / S))).

pv_cylinder_inner_diameter(I) <-
    pv_cylinder_thickness_elastic_failure(E), pv_cylinder_thickness_yield_failure(Y),
    I = diameter - 2 * max(E, Y).

pv_cylinder_material_volume(pi * length * T) <- 
    pv_cylinder_inner_diameter(I), T = (diameter / 2)^2 - (I / 2)^2.

pv_cylinder_weight(V * D * 1000) <- pv_cylinder_material_volume(V), density(material, D).

pv_endcap_thickness_elastic_failure(F) <-
    youngs_modulus(material, Y), poissons_ratio(material, R), crush_depth(D),
    F = diameter * sqrt(D / Y / (0.2 * 2 / (sqrt(3 * (1 - R^2))))).

pv_endcap_thickness_yield_failure(F) <-
    R = diameter / 2, crush_depth(D), yield_stress(material, S), 
    F = R - R * (1 - 1.5 * (D / S))^(1/3).

pv_endcap_inner_diameter(I) <-
    pv_endcap_thickness_elastic_failure(E), pv_endcap_thickness_yield_failure(Y),
    I = diameter - 2 * max(E, Y).

pv_endcap_material_volume(4 / 3 * pi * T) <-
    pv_endcap_inner_diameter(I), T = (diameter / 2)^3 - (I / 2)^3.

pv_endcap_weight(V * D * 1000) <- pv_endcap_material_volume(V), density(material, D).

pv_weight(E + C) <- pv_cylinder_weight(C), pv_endcap_weight(E).

# RANGE VS SPEED CALCULATIONS

# some relations from this section are computed elsewhere

fineness_ratio(R) <- R = length / diameter.
midbody_volume(V) <-
    pv_length(P), R = diameter / 2,
    V = P * pi * R * (4 / 3 * pi * radius^3).

battery_weight(W) <-
    pv_displacement(D), pv_weight(P),
    W = battery_bouyancy_fraction * (D - P).
battery_capacity(W * E) <- E = battery_specific_energy, battery_weight(W).

nose_displacement(D) <-
    R = diameter / 2, nose_length(N),
    D = density_of_seawater * (N - R) * pi * R^2.

pv_excess_bouyancy(D - P - W) <- pv_displacement(D), pv_weight(P), battery_weight(W).

# WARNING PANEL

pv_length_ok <- pv_length(L), L > 0.
battery_weight_ok <- battery_weight(W), W > 0.
fineness_ok <- fineness_ratio(R), R >= 5.5, R <= 7.5.
depth_ok <- depth_rating < 11,030.

all_ok <- pv_length_ok, battery_weight_ok, fineness_ok, depth_ok.

# RANGE TABLE COMPUTATIONS

# assume speed is given in m/s

# drag (D) at speed S
drag(S, D)
    total_wetted_area(A), F = 1 + appendage_added_area,
    D = A * drag * F * density_of_seawater * S^2 / 2.

# power (W) required to move at speed S
prop_power(S, W) <- drag(S, D), W = S * D / propulsion_efficiency.

# range (R) of LRAUV at speed S with hotel load W
range(S, W, R) <-
    battery_capacity(C), prop_power(S, P), 
    R = S * C * 3.6 / (W + P).

# the most efficient speed (S) given hotel load W
most_efficient_speed(W, S) <-
    total_wetted_area(A), F = 1 + appendage_added_area,
    D = drag_coefficient * total_wetted_area * F * density_of_seawater,
    S = propulsion_efficiency * W / D^(1/3).

# vehicle range (R) at the most efficient speed for hotel load W
range_at_most_efficient_speed(W, R) <-
    battery_capacity(C), most_efficient_speed(W, S),
    R = C / (1.5 * W) * S * 3.6

# EXAMPLE OPTIMIZATION TASK

# encodes the following task:
# 1. find values of the non-observed parameters
# 2. such that the constraint is satisfied and
# 3. the objective is maximized.

!observe
    depth_rating : 300,
    safety_factor : 1.5,
    battery_specific_energy : 360,
    battery_bouyancy_fraction : 0.5,
    hotel_power_draw : 20,
    drag_coefficient : 0.0079,
    appendage_added_area : 0.1,
    propulsion_efficiency : 0.5,
    density_of_seawater : 1027.

!constrain all_ok.

!optimize R : range_at_most_efficient_speed(8, R).