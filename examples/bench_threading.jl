# Speedup of the multithreaded single-operator rotation.
#
#     julia --project -t 1 examples/bench_threading.jl
#     julia --project -t 2 examples/bench_threading.jl
#     julia --project -t 4 examples/bench_threading.jl
#     julia --project -t 8 examples/bench_threading.jl
#
# Each run measures BOTH the serial path (`threaded=false`) and the threaded
# path in the SAME process, so the speedup column is self-contained and the
# rows are comparable across thread counts without worrying about machine
# state or compilation differences between processes.
#
# The operator is deliberately large: threading a rotation only pays once
# there are enough terms to outweigh task overhead and per-thread staging
# (see THREADED_ROTATE_MIN).

using PauliOperators
using Printf
using Random

const N       = 200          # UInt256 storage word
const NTERMS  = 200_000
const NTRIALS = 5            # repeat and take the best time
const THETA   = 0.21

function build_operator(N, nterms; seed = 7)
    Random.seed!(seed)
    O = PauliSum(N, ComplexF64)
    sizehint!(O, nterms)
    while length(O) < nterms
        O[rand(PauliBasis{N})] = (2 * rand() - 1) + (2 * rand() - 1) * im
    end
    return O
end

# Time ONE rotation on a fixed operator. Each trial starts from a fresh copy,
# so every trial does identical work -- chaining rotations instead would let
# the term count (and therefore the workload) grow between configurations and
# make the timings incomparable. The copy is made outside the timed region.
function time_rotation(O0, G, threaded::Bool, backend::Symbol)
    mk() = backend == :spv ? SparsePauliVector(O0) : deepcopy(O0)

    evolve_threaded!(mk(), G, THETA; threaded = threaded)   # warm up

    best = Inf
    nterms = 0
    for _ in 1:NTRIALS
        O = mk()
        t = @elapsed evolve_threaded!(O, G, THETA; threaded = threaded)
        best = min(best, t)
        nterms = length(O)
    end
    return best, nterms
end

@printf("\nThreaded rotation benchmark\n")
@printf("N=%d (word %s), %d terms, 1 rotation, best of %d\n", N, word_type(N), NTERMS, NTRIALS)
@printf("Threads.nthreads() = %d   THREADED_ROTATE_MIN = %d\n\n",
        Threads.nthreads(), THREADED_ROTATE_MIN)

O0 = build_operator(N, NTERMS)
G  = PauliBasis(Pauli(N; X = [1, 2]))

@printf("%-10s %12s %12s %10s %12s\n",
        "backend", "serial(s)", "threaded(s)", "speedup", "final terms")
@printf("%s\n", repeat("-", 62))
for backend in (:dict, :spv)
    t_ser, n_ser = time_rotation(O0, G, false, backend)
    t_thr, n_thr = time_rotation(O0, G, true,  backend)
    @assert n_ser == n_thr "threaded and serial disagree on term count"
    @printf("%-10s %12.3f %12.3f %9.2fx %12d\n",
            backend, t_ser, t_thr, t_ser / t_thr, n_thr)
end
println()
