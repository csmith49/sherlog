!namespace nn.

run(X ; model[X]) <- is_input(X).
loss(X, Y ; loss[X, Y]) <- run(X, Y).

is_input(day1_x).

!evidence loss(day1_x, day1_y, 0.0).