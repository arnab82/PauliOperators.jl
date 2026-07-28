using Random

# ============================================================
# Abstract Types
# ============================================================

"""
    TruncationStrategy

Abstract supertype for term-truncation strategies applied by `truncate!` and
the `truncation`/`local_truncation` keywords of `evolve!`. Define a new
strategy by subtyping and implementing `_apply!(O, s)`.
"""
abstract type TruncationStrategy end

"""
    CorrectionAccumulator

Abstract supertype for truncation-error trackers passed to `truncate!` and
`evolve!`: observables are measured before and after each truncation and the
differences accumulate. See `EnergyCorrection`, `EnergyVarianceCorrection`,
`NoCorrection`. Define a new accumulator by subtyping and implementing
`_measure(O, corr)` and `_accumulate!(corr, before, after)`.
"""
abstract type CorrectionAccumulator end


# ============================================================
# Truncation Strategy Types
# ============================================================

"""
    NoTruncation()

Identity truncation — does nothing.
"""
struct NoTruncation <: TruncationStrategy end

"""
    CoeffTruncation(thresh::Float64)

Remove Pauli terms with |coefficient| <= `thresh`.
"""
struct CoeffTruncation <: TruncationStrategy
    thresh::Float64
end
CoeffTruncation() = CoeffTruncation(1e-6)

"""
    WeightTruncation(max_weight::Int)

Remove Pauli terms with Pauli weight > `max_weight`.
"""
struct WeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    XWeightTruncation(max_weight::Int)

Remove Pauli terms with X-weight (number of X/Y factors) > `max_weight`.
"""
struct XWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    MajoranaWeightTruncation(max_weight::Int)

Remove Pauli terms with Majorana weight > `max_weight`.
"""
struct MajoranaWeightTruncation <: TruncationStrategy
    max_weight::Int
end

"""
    WeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with Pauli weight.
`alpha = 0` reduces to `CoeffTruncation(thresh)`; large `alpha` approaches
a hard weight cutoff.
"""
struct WeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
WeightDampedTruncation(alpha::Real) = WeightDampedTruncation(alpha, 1e-6)

"""
    XWeightDampedTruncation(alpha::Float64, thresh::Float64)

Remove Pauli terms with |coefficient|·exp(-alpha·x_weight) <= `thresh`,
i.e. a coefficient threshold that grows exponentially with X-weight (the
number of X/Y factors). `alpha = 0` reduces to `CoeffTruncation(thresh)`;
large `alpha` approaches a hard X-weight cutoff.
"""
struct XWeightDampedTruncation <: TruncationStrategy
    alpha::Float64
    thresh::Float64
end
XWeightDampedTruncation(alpha::Real) = XWeightDampedTruncation(alpha, 1e-6)

"""
    StochasticCoeffTruncation(epsilon::Float64; rng=Random.default_rng())

Unbiased stochastic compression (Russian Roulette). Wraps `stochastic_clip!`.

For each term with |c| < epsilon:
- Keep with probability |c|/epsilon (promote to epsilon·sign(c))
- Delete with probability 1 - |c|/epsilon
"""
struct StochasticCoeffTruncation <: TruncationStrategy
    epsilon::Float64
    rng::AbstractRNG
end
StochasticCoeffTruncation(epsilon::Float64) = StochasticCoeffTruncation(epsilon, Random.default_rng())

"""
    StochasticSamplingTruncation(n_keep::Int; rng=Random.default_rng())

Stochastically sample `n_keep` terms via importance sampling with probabilities
proportional to |c_i|^2. Kept terms are rescaled to preserve norm.
"""
struct StochasticSamplingTruncation <: TruncationStrategy
    n_keep::Int
    rng::AbstractRNG
end
StochasticSamplingTruncation(n_keep::Int) = StochasticSamplingTruncation(n_keep, Random.default_rng())

"""
    AdaptiveTruncation(max_terms::Int, min_thresh::Float64)

If the number of terms exceeds `max_terms`, increase the clipping threshold
to reduce the operator size. Otherwise clip at `min_thresh`.
"""
struct AdaptiveTruncation <: TruncationStrategy
    max_terms::Int
    min_thresh::Float64
end
AdaptiveTruncation(; max_terms::Int=10000, min_thresh::Float64=1e-12) = AdaptiveTruncation(max_terms, min_thresh)

"""
    CompositeTruncation(strategies...)

Apply multiple truncation strategies in sequence.

Strategies are stored as a typed `Tuple` rather than `Vector{TruncationStrategy}`,
so the per-element dispatches inside `_apply!` resolve at compile time and the
inner `coeff_clip!` / `weight_clip!` calls inline. Constructing via the variadic
form (`CompositeTruncation(CoeffTruncation(1e-4), WeightTruncation(5))`) is
the supported call style; an `AbstractVector` constructor is also provided
for convenience but converts to a tuple internally.
"""
struct CompositeTruncation{S<:Tuple} <: TruncationStrategy
    strategies::S
end
CompositeTruncation(s::TruncationStrategy...) = CompositeTruncation(s)
CompositeTruncation(v::AbstractVector{<:TruncationStrategy}) = CompositeTruncation(Tuple(v))


# ============================================================
# _apply! — raw truncation dispatch (internal)
# ============================================================

function _apply!(O::PauliSum{N}, ::NoTruncation) where N
    return O
end

function _apply!(O::PauliSum{N}, s::CoeffTruncation) where N
    return coeff_clip!(O, s.thresh)
end

function _apply!(O::PauliSum{N}, s::WeightTruncation) where N
    return weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::XWeightTruncation) where N
    return x_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::MajoranaWeightTruncation) where N
    return majorana_weight_clip!(O, s.max_weight)
end

function _apply!(O::PauliSum{N}, s::WeightDampedTruncation) where N
    return weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::XWeightDampedTruncation) where N
    return x_weight_damped_clip!(O, s.alpha, s.thresh)
end

function _apply!(O::PauliSum{N}, s::StochasticCoeffTruncation) where N
    return stochastic_clip!(O, s.epsilon; rng=s.rng)
end

function _apply!(O::PauliSum{N}, s::StochasticSamplingTruncation) where N
    length(O) <= s.n_keep && return O

    keys_vec = collect(keys(O))
    weights = [abs2(O[k]) for k in keys_vec]
    norm_sq = sum(weights)
    sampling_keys = [rand(s.rng)^(1.0/w) for w in weights]

    kept_idx = partialsortperm(sampling_keys, 1:s.n_keep, rev=true)
    kept_set = Set(keys_vec[i] for i in kept_idx)

    kept_norm_sq = sum(abs2(O[k]) for k in kept_set)
    filter!(p -> p.first in kept_set, O)

    if kept_norm_sq > 0
        scale = sqrt(norm_sq / kept_norm_sq)
        for k in keys(O)
            O[k] *= scale
        end
    end

    return O
end

function _apply!(O::PauliSum{N}, s::AdaptiveTruncation) where N
    if length(O) > s.max_terms
        coeffs = sort(abs.(collect(values(O))))
        if length(coeffs) > s.max_terms
            thresh = coeffs[end - s.max_terms]
            coeff_clip!(O, thresh)
        end
    else
        coeff_clip!(O, s.min_thresh)
    end
    return O
end

# Recursive tail-pop iteration over the heterogeneous tuple of strategies so
# each `_apply!(O, strategy)` resolves at compile time and inlines.
@inline _apply_tup!(O, ::Tuple{}) = O
@inline _apply_tup!(O, s::Tuple)  = (_apply!(O, first(s)); _apply_tup!(O, Base.tail(s)))

function _apply!(O::PauliSum{N}, s::CompositeTruncation) where N
    _apply_tup!(O, s.strategies)
    return O
end


# ============================================================
# Correction Accumulator Types
# ============================================================

"""
    NoCorrection()

Track nothing during truncation. Zero overhead.
"""
struct NoCorrection <: CorrectionAccumulator end

"""
    EnergyCorrection(ψ::Ket{N})

Track accumulated change in ⟨ψ|O|ψ⟩ due to truncation.
"""
mutable struct EnergyCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
end
EnergyCorrection(ψ::Ket{N}) where N = EnergyCorrection{N}(ψ, 0.0)

"""
    EnergyVarianceCorrection(ψ::Ket{N})

Track accumulated changes in both ⟨ψ|O|ψ⟩ and Var(O,ψ) due to truncation.
"""
mutable struct EnergyVarianceCorrection{N} <: CorrectionAccumulator
    ψ::Ket{N}
    accumulated_energy::Float64
    accumulated_variance::Float64
end
EnergyVarianceCorrection(ψ::Ket{N}) where N = EnergyVarianceCorrection{N}(ψ, 0.0, 0.0)


# ============================================================
# Single-pass truncation deltas (fast corrections)
# ============================================================
# For pure-drop truncations (everything compilable to a MergeFilter), the
# corrections can be computed exactly from the dropped terms alone: writing
# O = A + B with B the dropped part,
#
#     Δ⟨O⟩  = -⟨B⟩
#     ΔVar  = -( Var(B) + 2·cov(A,B) ),  cov(A,B) = Re⟨a|b⟩ - ⟨A⟩⟨B⟩
#
# where b = B|ψ⟩ lives on at most (#dropped) kets and a = A|ψ⟩ is only ever
# needed on b's support. So instead of the two full ⟨O²⟩ evaluations of the
# before/after `_measure` route (each builds an O(len(O)) KetSum), the exact
# correction costs O(#dropped) at the drop sites plus ONE sweep of the kept
# terms with lookups into a (#dropped)-entry dict.
#
# `TruncationDelta` is the drop sink: kernels call `_sink_drop!` on each
# net-merged coefficient the filter rejects, and `_finalize_delta!` folds the
# result into the correction accumulator. Ket bits are keyed as Int128
# (`% Int128` reinterprets the SPV's unsigned words losslessly).

mutable struct TruncationDelta
    ψv::Int128                      # reference ket bits
    b::Dict{Int128,ComplexF64}      # B|ψ⟩ amplitudes, keyed by ket bits
end
TruncationDelta(ψ::Ket) = TruncationDelta(ψ.v, Dict{Int128,ComplexF64}())

# apply the (z,x) Pauli word to the ket bits kv: same math as
# Base.:*(::PauliBasis, ::Ket) in multiplication.jl, on raw words
@inline function _ket_action(z::Int128, x::Int128, kv::Int128)
    kv2 = x ⊻ kv
    idx = ((4 - count_ones(z & x) % 4) % 4 + 2 * (count_ones(z & kv2) % 2)) % 4 + 1
    return PHASE_TBL[idx], kv2
end

@inline _sink_drop!(::Nothing, z, x, c) = nothing
@inline function _sink_drop!(Δ::TruncationDelta, z, x, c)
    ph, kv2 = _ket_action(z % Int128, x % Int128, Δ.ψv)
    Δ.b[kv2] = get(Δ.b, kv2, zero(ComplexF64)) + ph * c
    return nothing
end

# amplitudes of the KEPT operator on b's support (+ the reference ket)
function _sweep_kept!(adict::Dict{Int128,ComplexF64}, aψ::Base.RefValue{ComplexF64},
                      v::SparsePauliVector{N,W,T}, ψv::Int128) where {N,W,T}
    @inbounds for i in 1:v.n
        ph, kv2 = _ket_action(v.z[i] % Int128, v.x[i] % Int128, ψv)
        hit = haskey(adict, kv2)
        (hit || kv2 == ψv) || continue
        amp = ph * v.c[i]
        kv2 == ψv && (aψ[] += amp)
        hit && (adict[kv2] += amp)
    end
    return nothing
end

function _sweep_kept!(adict::Dict{Int128,ComplexF64}, aψ::Base.RefValue{ComplexF64},
                      O::PauliSum{N}, ψv::Int128) where {N}
    for (p, c) in O
        ph, kv2 = _ket_action(p.z, p.x, ψv)
        hit = haskey(adict, kv2)
        (hit || kv2 == ψv) || continue
        amp = ph * c
        kv2 == ψv && (aψ[] += amp)
        hit && (adict[kv2] += amp)
    end
    return nothing
end

_finalize_delta!(::NoCorrection, Δ::TruncationDelta, O) = nothing

function _finalize_delta!(corr::EnergyCorrection, Δ::TruncationDelta, O)
    corr.accumulated_energy += -real(get(Δ.b, Δ.ψv, zero(ComplexF64)))
    return nothing
end

function _finalize_delta!(corr::EnergyVarianceCorrection, Δ::TruncationDelta, O)
    isempty(Δ.b) && return nothing
    eB = get(Δ.b, Δ.ψv, zero(ComplexF64))       # ⟨ψ|B|ψ⟩ (complex-safe)
    bb = 0.0                                    # ⟨Bψ|Bψ⟩
    for amp in values(Δ.b)
        bb += abs2(amp)
    end
    adict = Dict{Int128,ComplexF64}()
    for k in keys(Δ.b)
        adict[k] = zero(ComplexF64)
    end
    aψ = Ref(zero(ComplexF64))
    _sweep_kept!(adict, aψ, O, Δ.ψv)
    ab = zero(ComplexF64)                       # ⟨Aψ|Bψ⟩ (only b's support)
    for (k, bv) in Δ.b
        ab += conj(adict[k]) * bv
    end
    eA = aψ[]                                   # ⟨ψ|A|ψ⟩
    # matches variance() = real(‖Oψ‖² − ⟨ψ|O|ψ⟩²) exactly:
    #   Δ‖Oψ‖² = −(bb + 2Re⟨a|b⟩),  Δ⟨O⟩² = −(2·eA·eB + eB²)
    corr.accumulated_energy   += -real(eB)
    corr.accumulated_variance += -(bb + 2 * real(ab)) + real(2 * eA * eB + eB * eB)
    return nothing
end


# ============================================================
# measure — snapshot quantities before/after truncation
# ============================================================

_measure(::AnyPauliSum, ::NoCorrection) = nothing

function _measure(O::AnyPauliSum{N}, corr::EnergyCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),)
end

function _measure(O::AnyPauliSum{N}, corr::EnergyVarianceCorrection{N}) where N
    return (energy = real(expectation_value(O, corr.ψ)),
            variance = real(variance(O, corr.ψ)))
end


# ============================================================
# _accumulate! — update accumulator with before/after diffs
# ============================================================

_accumulate!(::NoCorrection, before, after) = nothing

function _accumulate!(corr::EnergyCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
end

function _accumulate!(corr::EnergyVarianceCorrection, before, after)
    corr.accumulated_energy += after.energy - before.energy
    corr.accumulated_variance += after.variance - before.variance
end


# ============================================================
# truncate! — unified entry point
# ============================================================

"""
    truncate!(O::PauliSum, strategy::TruncationStrategy,
              corr::CorrectionAccumulator=NoCorrection())

Apply `strategy` to truncate `O` in-place. If a `CorrectionAccumulator` is
provided, measure quantities before and after truncation and accumulate the
differences.

Users can define new strategies by subtyping `TruncationStrategy` and
implementing `_apply!(O, s)`. New correction types are defined by subtyping
`CorrectionAccumulator` and implementing `_measure(O, corr)` and
`_accumulate!(corr, before, after)`.
"""
function truncate!(O::AnyPauliSum, strategy::TruncationStrategy,
                   corr::CorrectionAccumulator=NoCorrection())
    # Fast path: pure-drop (compilable) strategies with the built-in
    # accumulators use the single-pass delta formula instead of full
    # before/after measurements. Strategies that rescale surviving
    # coefficients (e.g. StochasticSamplingTruncation) are not pure drops and
    # take the measure-based fallback below.
    if corr isa Union{EnergyCorrection,EnergyVarianceCorrection} &&
       !(strategy isa NoTruncation) && _is_compilable(strategy)
        Δ = TruncationDelta(corr.ψ)
        _apply_dropping!(O, _compile_filter(strategy), Δ)
        _finalize_delta!(corr, Δ, O)
        return O
    end
    before = _measure(O, corr)
    _apply!(O, strategy)
    after = _measure(O, corr)
    _accumulate!(corr, before, after)
    return O
end

# filter with drop-sink observation (pure-drop strategies only)
function _apply_dropping!(O::PauliSum{N}, f, Δ::TruncationDelta) where {N}
    filter!(O) do pr
        p, c = pr
        if should_drop(f, p.z % UInt128, p.x % UInt128, abs(c))
            _sink_drop!(Δ, p.z, p.x, c)
            return false
        end
        return true
    end
    return O
end
_apply_dropping!(v::SparsePauliVector, f, Δ::TruncationDelta) = _compact_spv!(v, f, Δ)
