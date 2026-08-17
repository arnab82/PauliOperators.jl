# ============================================================
# Multithreaded Heisenberg-picture rotation of a single operator.
#
# Single node, shared memory, no worker processes. A rotation splits every
# term that anticommutes with the generator G:
#
#     O(θ) = cos(θ)·O - i·sin(θ)·G·O
#
# The cos part rescales a term in place; the sin part creates a NEW term G·p.
# Both halves are per-term independent, so the sweep over terms is chunked
# across Julia threads. Each thread
#
#   * rescales its own chunk's coefficients in place (disjoint indices, so no
#     synchronisation), and
#   * appends its sin-branch terms to its OWN staging buffer (no shared
#     container, so no locking and no races).
#
# The staged terms are then concatenated and merged into the operator by the
# ordinary serial merge, which restores the sorted duplicate-free invariant.
# Merging is serial by construction: sin branches from different chunks can
# collide on the same PauliBasis and must sum.
#
# Threading only pays once a rotation has enough terms to outweigh the
# @spawn/@sync overhead and the staging allocation; below
# `THREADED_ROTATE_MIN` the serial kernel is used instead. `evolve_threaded!`
# is otherwise a drop-in for `evolve!` and is verified against it.
# ============================================================

"""
    THREADED_ROTATE_MIN

Term count below which `evolve_threaded!` falls back to the serial kernel.
Splitting a small rotation costs more in task overhead and per-thread staging
than it saves.
"""
const THREADED_ROTATE_MIN = 1 << 12

# contiguous chunk ranges partitioning 1:n into k parts (trailing parts may be
# empty when k > n)
function _chunk_ranges(n::Int, k::Int)
    sz = cld(n, k)
    return [((c - 1) * sz + 1):min(c * sz, n) for c in 1:k]
end

# How many threads to actually use for a rotation over `n` terms.
@inline function _rotate_nthreads(n::Int, threaded::Bool)
    (threaded && Threads.nthreads() > 1 && n >= THREADED_ROTATE_MIN) || return 1
    return min(Threads.nthreads(), n)
end

# ------------------------------------------------------------
# SparsePauliVector
# ------------------------------------------------------------

# Rotate one contiguous slice of the live buffer. Writes only to O.c[range]
# (its own chunk) and to `stage` (its own buffer).
function _rotate_chunk_threaded!(O::SparsePauliVector{N,W,T}, range,
                                 gz::W, gx::W, ng::Int,
                                 cosθ::Float64, sinθ::Float64,
                                 stage::Vector{Tuple{W,W,T}}) where {N,W,T}
    @inbounds for i in range
        zi = O.z[i]
        xi = O.x[i]
        # anticommute test: G p = -p G iff the symplectic form is odd
        m1 = count_ones(gx & zi)
        m2 = count_ones(gz & xi)
        iseven(m1 - m2) && continue

        zp = gz ⊻ zi
        xp = gx ⊻ xi
        k = (count_ones(zp & xp) - ng - count_ones(zi & xi) + 2 * m1) & 3
        cnew = (T(k - 2) * sinθ) * O.c[i]
        O.c[i] *= cosθ
        push!(stage, (zp, xp, cnew))
    end
    return nothing
end

"""
    evolve_threaded!(O::SparsePauliVector{N,W}, G::PauliBasis{N,W}, θ; threaded=true)

In-place `exp(iθ/2 G) O exp(-iθ/2 G)`, with the rotation sweep chunked across
Julia threads. Same result as [`evolve!`](@ref) — verified against it — and
falls back to it when the operator is small or only one thread is available.

Start Julia with `--threads=N` for this to do anything.
"""
function evolve_threaded!(O::SparsePauliVector{N,W,T}, G::PauliBasis{N,W}, θ::Real;
                          threaded::Bool = true) where {N,W,T}
    nt = _rotate_nthreads(O.n, threaded)
    nt == 1 && return evolve!(O, G, θ)

    gz, gx = _pack(W, G)
    ng = count_ones(gz & gx)
    cosθ = cos(θ)
    sinθ = sin(θ)

    ranges = _chunk_ranges(O.n, nt)
    stages = [Vector{Tuple{W,W,T}}() for _ in 1:nt]
    # a rotation creates at most one new term per existing term
    for c in 1:nt
        sizehint!(stages[c], length(ranges[c]))
    end

    @sync for c in 1:nt
        Threads.@spawn _rotate_chunk_threaded!(O, ranges[c], gz, gx, ng,
                                               cosθ, sinθ, stages[c])
    end

    # Concatenate the per-thread staging into the merge workspace, then merge
    # serially: sin branches from different chunks can land on the same basis.
    m = sum(length, stages)
    m == 0 && return O
    length(O.ws) < m && resize!(O.ws, m)
    i = 0
    @inbounds for st in stages, tr in st
        i += 1
        O.ws[i] = tr
    end
    _sort_ws!(O.ws, 1, m)
    _merge_spv!(O, m, NOFILTER)
    return O
end

# ------------------------------------------------------------
# Dict-backed PauliSum
# ------------------------------------------------------------
# A Dict cannot take concurrent writes, so the threaded pass is read-only over
# O: each thread records which of its keys anticommute (to be cos-scaled
# afterwards) and stages the sin branches. Both the rescale and the insert run
# serially after the join. The parallel part is the anticommutation test and
# the Pauli product, which is where the work is.

function _rotate_chunk_threaded_dict!(O::PauliSum{N,W,T}, ks, range, G, _sin,
                                      stage::Vector{Tuple{PauliBasis{N,W},T}},
                                      coskeys::Vector{PauliBasis{N,W}}) where {N,W,T}
    @inbounds for idx in range
        p = ks[idx]
        commute(p, G) && continue
        tmp = O[p] * _sin * G * p
        push!(stage, (PauliBasis(tmp), convert(T, coeff(tmp))))
        push!(coskeys, p)
    end
    return nothing
end

"""
    evolve_threaded!(O::PauliSum{N,W}, G::PauliBasis{N,W}, θ; threaded=true)

Dict-backed counterpart. The anticommutation scan and Pauli products run in
parallel; the coefficient updates and insertions are applied serially after
the join, because `Dict` is not safe for concurrent writes. Expect less
speedup than the `SparsePauliVector` path — prefer SPV for threaded work.
"""
function evolve_threaded!(O::PauliSum{N,W,T}, G::PauliBasis{N,W}, θ::Real;
                          threaded::Bool = true) where {N,W,T}
    nt = _rotate_nthreads(length(O), threaded)
    nt == 1 && return evolve!(O, G, θ)

    _cos = cos(θ)
    _sin = 1im * sin(θ)
    ks = collect(keys(O))

    ranges = _chunk_ranges(length(ks), nt)
    stages  = [Vector{Tuple{PauliBasis{N,W},T}}() for _ in 1:nt]
    coskeys = [Vector{PauliBasis{N,W}}() for _ in 1:nt]

    @sync for c in 1:nt
        Threads.@spawn _rotate_chunk_threaded_dict!(O, ks, ranges[c], G, _sin,
                                                    stages[c], coskeys[c])
    end

    for c in 1:nt, p in coskeys[c]
        O[p] = O[p] * _cos
    end
    for c in 1:nt, (pb, cval) in stages[c]
        O[pb] = get(O, pb, zero(T)) + cval
    end
    return O
end

"""
    evolve_threaded!(O, generators, angles; threaded=true, truncation=NoTruncation())

Apply a sequence of rotations, optionally truncating after each one. Works on
either backend.
"""
function evolve_threaded!(O::AnyPauliSum{N,W,T},
                          generators::Vector{<:PauliBasis{N,W}},
                          angles::Vector{<:Real};
                          threaded::Bool = true,
                          truncation::TruncationStrategy = NoTruncation()) where {N,W,T}
    length(generators) == length(angles) ||
        throw(DimensionMismatch("generators and angles must have the same length"))
    for (G, θ) in zip(generators, angles)
        evolve_threaded!(O, G, θ; threaded = threaded)
        truncation isa NoTruncation || truncate!(O, truncation)
    end
    return O
end
