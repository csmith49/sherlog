swap_rate(X, Y; swap_nn[X, Y]).
swap(X, Y; categorical[R]) <- swap_rate(X, Y, R).

quicksort([], []).
quicksort(X :: XS, YS) <-
    partition(XS, X, Left, Right),
    quicksort(Left, LS),
    quicksort(Right, RS),
    append(LS, X :: RS, YS).

partition([], Y, [], []).
partition(X :: XS, Y, X :: LS, RS) <- swap(X, Y, 0), partition(XS, Y, LS, RS).
partition(X :: XS, Y, LS, X :: RS) <- swap(X, Y, 1), partition(XS, Y, LS, RS).

append([], YS, YS).
append(X :: XS, YS, X :: ZS) <- append(XS, YS, ZS).

forth_sort(I, O) <- quicksort(I, O).

!evidence forth_sort([0, 1], [0, 1]).