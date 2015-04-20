# python-logprobs

It is very common to use log probabilities in statistics, machine learning, and scientific computation, for example to avoid underflowing floats when calculating likelihood functions.  Most of the time, the math is very simple, and doing things "by hand" is fine.  But sometimes the numerics are tricky, and it seems like there should be a standard(ish) way of doing these computations, where the optimization/numerics are behind a nice abstraction layer, and all you have to do is call functions that do what you want.

This module is supposed to abstract away the (simple) math needed to do arithmetic with log probabilities.  For example, If `p` and `q` are probabilities, then `x` and `y` are representations in log space:
```code
x = log(p) and y = log(q)  <==>  p = e^x and q = e^y
```
To multiply two probabilities, `r = p*q`, it's easy to stay in the log space:
```
w = log(r) = log(p*q) = log(e^x * e^y) = log(e^(x + y)) = x + y
```
So the function `multiply_log_weights` assumes you already have `x` and `y`, the log space representations of some probabilities, and returns the log space result of multiplying the probabilities, i.e. adding the log space values. To get back to normal space, just exponentiate using `convert_logweight_to_standard(w)`.  If you don't have `x` to start with, but only `p`, you gan get it using `convert_standard_to_logweight(p)`, which simply takes a log.

Currently, the only non-obvious functions in the module are adding and subtracting log probabilities.
