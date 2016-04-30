nnetObj <- function(sizes) {
  structure(list(sizes = sizes,
                 num_layers = length(sizes),
                 sigmoid = function(v) 1 / (1 + exp(-v)),
                 sigmoid_prime = function(v) (1 / (1 + exp(-v))) * (1 - 1 / (1 + exp(-v))),
                 cost = function(a, y) 0.5 * apply((a - y)^2,2,sum),
                 cost_derivative = function(activation, y) (activation - y),
                 biases = lapply(sizes[-1], rnorm),
                 weights = mapply(randMat, sizes[-1], sizes[-length(sizes)]), SIMPLIFY=FALSE),
            class="net")
}