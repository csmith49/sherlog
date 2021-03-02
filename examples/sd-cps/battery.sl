# encoding attributes for, e.g., a D battery

diameter("D", 34.2).
length("D", 61.5).
volume("D", 5190).
voltage("D", 2.5).
watt_hours("D", 40).

# encoding voltage and capacity of packing configurations
# packing configuration (B, S, P) contains:
# 1. battery of type B
# 2. S batteries in series per-layer
# 3. P layers in parallel

packing_voltage(B, S, V) <- voltage(B, V'), V = S * V'.
packing_energy(B, S, P, E) <- watt_hours(B, W), E = N * W, N = S * P.

# computing minimal hull inner diameter to satisfy a packing configuration
# complex combinatorial problem dependent on:
# 1. diameter of battery used
# 2. the number of batteries per-layer
# instead of computing directly, rely on table lookup from www.packomania.com

layer_diameter(B, S; packomania[D, S]) <- diameter(B, D).

# assign domains to parameters of interest

!parameter battery  : Discrete["D", "C", "A", "AA", "AAA"].
!parameter series   : Natural[0, 100].
!parameter parallel : Natural[0, 100].

# and, finally, encode the mission constraints

!optimize:
    packing_voltage(battery, series, V), V >= 24,
    packing_energy(battery, series, E), E >= 8000,
    layer_diameter(battery, series, D), D <= 10.