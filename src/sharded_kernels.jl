# The sharded workspace sorter is NOT bitint's `_sort_ws!`: the two are textually
# identical but call different comparators. bitint's sorts x-major (its fused
# truncation corrections need x-runs contiguous); the sharded merge assumes the
# z-major order it was written against. Sharing one sorter silently corrupts the
# merge, so the sharded engine keeps its own.
function _shard_sort_ws!(ws::Vector{TT}, lo::Int, hi::Int) where {TT<:Tuple}
    @inbounds while hi - lo >= 24
        mid = (lo + hi) >>> 1
        if _shard_key_lt(ws[mid], ws[lo])
            ws[mid], ws[lo] = ws[lo], ws[mid]
        end
        if _shard_key_lt(ws[hi], ws[mid])
            ws[hi], ws[mid] = ws[mid], ws[hi]
            if _shard_key_lt(ws[mid], ws[lo])
                ws[mid], ws[lo] = ws[lo], ws[mid]
            end
        end
        pivot = ws[mid]
        i, j = lo, hi
        while i <= j
            while _shard_key_lt(ws[i], pivot)
                i += 1
            end
            while _shard_key_lt(pivot, ws[j])
                j -= 1
            end
            if i <= j
                ws[i], ws[j] = ws[j], ws[i]
                i += 1
                j -= 1
            end
        end
        if j - lo < hi - i
            _shard_sort_ws!(ws, lo, j)
            lo = i
        else
            _shard_sort_ws!(ws, i, hi)
            hi = j
        end
    end
    @inbounds for k in lo+1:hi
        v = ws[k]
        m = k - 1
        while m >= lo && _shard_key_lt(v, ws[m])
            ws[m+1] = ws[m]
            m -= 1
        end
        ws[m+1] = v
    end
    return ws
end

# ============================================================
# Sharded engine kernels: rotation sweep, append sort, sorted merge.
#
# These are the steady-state hot path and MUST allocate zero bytes — the
# test suite enforces this with @ballocated. Rules: isbits arguments and
# plain Vectors only, no closures, no strings, no dynamic dispatch.
# ============================================================

# Lexicographic (z, x) key order, z-major. NOTE: deliberately DIFFERENT from
# bitint's `_key_lt`, which is x-major because the fused truncation corrections
# in truncation.jl need terms sharing an x-string to be contiguous. The sharded
# engine has no such requirement and keeps the z-major order it was written and
# tested against, so both orderings coexist under distinct names.
# Lexicographic (z, x) key order, z-major. Works on 2-tuples and on the
# (z, x, c) workspace triples (compares the first two fields).
@inline _shard_key_lt(a::Tuple, b::Tuple) =
    (a[1] < b[1]) | ((a[1] == b[1]) & (a[2] < b[2]))
@inline _shard_key_eq(a::Tuple, b::Tuple) = (a[1] == b[1]) & (a[2] == b[2])

# ------------------------------------------------------------
# Truncation filter, compiled once per evolve! from a TruncationStrategy —
# an isbits predicate evaluated per term with no dynamic dispatch.
# ------------------------------------------------------------

"""
    MergeFilter

Compiled truncation predicate for the sharded kernels. Sentinels disable
individual checks: `typemax(Int)` for the weight cutoffs, negative
thresholds for the coefficient cutoffs (`thresh = -1.0` keeps exact zeros,
matching `NoTruncation`; `coeff_clip!` semantics are "drop |c| <= thresh").
Built from a `TruncationStrategy` by `_compile_filter`.
"""


# Branchless-suffix-parity Majorana weight on packed words; the word-level
# analogue of `majorana_weight(::PauliBasis)` (see clip.jl for the derivation).
# `_majorana_weight_bits` comes from bitint's helpers.jl: its shift cascade is
# driven by `8*sizeof(W)`, so it stays correct at UInt256/512/1024. The shared1
# copy hardcoded shifts up to 64 and would silently mis-weight above UInt128.

# `should_drop` is bitint's (spv_kernels.jl) -- identical definition, kept in one place.

# `_compile_filter` is bitint's (spv_kernels.jl) -- identical definition, kept in one place.
# All `_compile_filter` methods (including CompositeTruncation) are bitint's,
# from spv_kernels.jl -- byte-identical apart from an error string, and bitint's
# set is a superset (it also handles NoTruncation).

# ------------------------------------------------------------
# Rotation kernel
# ------------------------------------------------------------

"""
    _shard__rotate_range!(z, x, c, lo, hi, gz, gx, n_g, cosθ, sinθ,
                   dz, dx, dc, cur, seg_end, f) -> (cur, created, overflowed)

Sweep terms `lo:hi` of one buffer under the rotation `exp(iθG)`: commuting
terms untouched; anticommuting terms cos-scaled in place, with the sin
branch (bits `G ⊻ P`, sign `i·i^k = ±1` computed purely from bits — the
fused-phase identity from commutator.jl) appended at `cur` in the
destination arrays unless the local filter drops it. `n_g` is
`count_ones(gz & gx)`, precomputed once per rotation.

Zero-allocation hot path. `overflowed` only fires if the driver's capacity
precheck was skipped or wrong; the driver treats it as an error.
"""
@inline function _shard__rotate_range!(z::Vector{W}, x::Vector{W}, c::Vector{T},
                                lo::Int, hi::Int,
                                gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                                dz::Vector{W}, dx::Vector{W}, dc::Vector{T},
                                cur::Int, seg_end::Int,
                                f::MergeFilter) where {W<:Unsigned, T<:Number}
    created = 0
    overflowed = false
    @inbounds for i in lo:hi
        zi = z[i]
        xi = x[i]
        m1 = count_ones(gx & zi)
        m2 = count_ones(gz & xi)
        iseven(m1 - m2) && continue
        zp = gz ⊻ zi
        xp = gx ⊻ xi
        k = (count_ones(zp & xp) - n_g - count_ones(zi & xi) + 2 * m1) & 3
        cnew = (T(k - 2) * sinθ) * c[i]
        c[i] *= cosθ
        should_drop(f, zp, xp, abs(cnew)) && continue
        if cur > seg_end
            overflowed = true
            continue
        end
        dz[cur] = zp
        dx[cur] = xp
        dc[cur] = cnew
        cur += 1
        created += 1
    end
    return cur, created, overflowed
end

"""
    _rotate_shard!(S, k, t, s_G, gz, gx, n_g, cosθ, sinθ, f) -> (created, overflowed)

Rotate shard `k` as thread `t`: sweep its live buffer and every append
segment up to the owner-snapshotted `sweep_hi`, appending sin branches into
thread `t`'s segment of the partner shard `k ⊻ s_G` (destination resolved
once — all of a shard's sin branches share one partner; that is the
rank-map property). Appends land at `cur ≥ sweep_hi`, so swept and written
ranges never overlap, and `sweep_hi` never moves mid-rotation (it is
snapshotted in the barrier-protected precheck phase, unlike the live
cursors of other threads).
"""
function _rotate_shard!(S::ShardedPauliSum{N,W,T}, k::Int, t::Int, s_G::Int,
                        gz::W, gx::W, n_g::Int, cosθ::Float64, sinθ::Float64,
                        f::MergeFilter) where {N,W,T}
    src = S.shards[k]
    j = ((k - 1) ⊻ s_G) + 1
    dst = S.shards[j]
    curt = S.cur[t]
    cur = curt[j]
    seg_end = dst.seg_lo[t+1] - 1
    created = 0
    overflowed = false

    cur, cr, ov = _shard__rotate_range!(src.z, src.x, src.c, 1, src.n,
                                 gz, gx, n_g, cosθ, sinθ,
                                 dst.az, dst.ax, dst.ac, cur, seg_end, f)
    created += cr
    overflowed |= ov
    @inbounds for seg in 1:S.nthreads
        lo = src.seg_lo[seg]
        hi = src.sweep_hi[seg] - 1
        cur, cr, ov = _shard__rotate_range!(src.az, src.ax, src.ac, lo, hi,
                                     gz, gx, n_g, cosθ, sinθ,
                                     dst.az, dst.ax, dst.ac, cur, seg_end, f)
        created += cr
        overflowed |= ov
    end
    curt[j] = cur
    return created, overflowed
end

"""
Snapshot shard `k`'s sweep bounds from the (currently stable) append
cursors, then check worst-case capacity: every swept term could anticommute
and append to the partner segment. MUST be called only by `k`'s owner,
in a phase where no thread is rotating (cursors quiescent).
"""
function _snapshot_and_precheck!(S::ShardedPauliSum, k::Int, t::Int, s_G::Int)
    src = S.shards[k]
    swept = src.n
    @inbounds for seg in 1:S.nthreads
        hi = S.cur[seg][k]
        src.sweep_hi[seg] = hi
        swept += hi - src.seg_lo[seg]
    end
    swept == 0 && return true
    j = ((k - 1) ⊻ s_G) + 1
    dst = S.shards[j]
    free = dst.seg_lo[t+1] - S.cur[t][j]
    return swept <= free
end

# ------------------------------------------------------------
# Sort + merge (window boundary)
# ------------------------------------------------------------

"""
Gather shard `j`'s pending append segments (each up to its cursor) into the
workspace as (z, x, c) triples. Returns the count. Allocation-free.
"""
function _gather_append!(sh::Shard{W,T}, cur::Vector{Vector{Int}}, j::Int,
                         nsegs::Int) where {W,T}
    m = 0
    @inbounds for t in 1:nsegs
        for i in sh.seg_lo[t]:(cur[t][j] - 1)
            m += 1
            sh.ws[m] = (sh.az[i], sh.ax[i], sh.ac[i])
        end
    end
    return m
end

"""
In-place quicksort (median-of-3, insertion sort below 24, recurse-smaller /
iterate-larger) of tuples by their first two fields — the (z, x) key for
merge-workspace triples, or (population, shard) pairs for rebalancing.
Hand-rolled because Base's default QuickSort allocates scratch; this is the
swap point for a future radix sort. Allocation-free.
"""

"""
    _merge_shard!(sh, m, f) -> (n_in, n_out)

Two-pointer merge of the sorted live buffer with `m` sorted workspace
triples into scratch, summing coefficients of equal keys (live first, then
appends in sorted order — equal-key *runs* in the appends are possible
across rotations of one window), dropping outputs the strict filter
rejects, then swapping scratch and live (pointer swap). Restores the
sorted, duplicate-free live invariant. Allocation-free.
"""
function _merge_shard!(sh::Shard{W,T}, m::Int, f::MergeFilter) where {W,T}
    n = sh.n
    z, x, c = sh.z, sh.x, sh.c
    sz, sx, sc = sh.sz, sh.sx, sh.sc
    ws = sh.ws
    cap = length(sz)
    out = 0
    i = 1
    j = 1
    @inbounds while i <= n || j <= m
        local kz::W, kx::W
        local acc::T
        if j > m || (i <= n && !_shard_key_lt(ws[j], (z[i], x[i])))
            kz = z[i]
            kx = x[i]
            acc = c[i]
            i += 1
        else
            kz, kx, acc = ws[j]
            j += 1
        end
        while j <= m && _shard_key_eq(ws[j], (kz, kx))
            acc += ws[j][3]
            j += 1
        end
        should_drop(f, kz, kx, abs(acc)) && continue
        out += 1
        out <= cap || error("shard live capacity exceeded during merge; " *
                            "raise capacity_factor or tighten truncation")
        sz[out] = kz
        sx[out] = kx
        sc[out] = acc
    end
    sh.z, sh.sz = sz, z
    sh.x, sh.sx = sx, x
    sh.c, sh.sc = sc, c
    sh.n = out
    return n + m, out
end

# ------------------------------------------------------------
# Boundary utilities: in-place compaction, coefficient histograms,
# expectation values (all allocation-free, owner-local)
# ------------------------------------------------------------

# In-place filter of a shard's live buffer (order-preserving, so the sorted
# invariant survives). Used for re-clips after an adaptive threshold raise
# and for the cold-path _apply!.
function _compact_shard!(sh::Shard{W,T}, f::MergeFilter) where {W,T}
    out = 0
    @inbounds for i in 1:sh.n
        should_drop(f, sh.z[i], sh.x[i], abs(sh.c[i])) && continue
        out += 1
        if out != i
            sh.z[out] = sh.z[i]
            sh.x[out] = sh.x[i]
            sh.c[out] = sh.c[i]
        end
    end
    sh.n = out
    return sh
end

# |c| exponent histogram over the live buffer: bin b holds coefficients
# with 2^(b-61) <= |c| < 2^(b-60), i.e. exponent(|c|) clamped to [-60, 3].
# `_hist_bin` is bitint's (spv_kernels.jl) -- identical, kept in one place.

function _hist_shard!(hist::Vector{Int}, sh::Shard{W,T}) where {W,T}
    @inbounds for i in 1:sh.n
        a = abs(sh.c[i])
        a == 0.0 && continue
        hist[_hist_bin(a)] += 1
    end
    return sh.n
end

# Largest coefficient threshold (a bin edge) keeping at most max_terms
# terms: everything in the returned threshold's bin and below is dropped.

# ⟨ψ|·|ψ⟩ over the shard's live buffer AND pending appends (the pre-merge
# state is live + appends with duplicates unmerged; expectation is linear,
# so summing them is exact). Computational-basis kets only: a term
# contributes c · (-1)^popcount(z & ψ) iff x == 0.
function _expectation_shard(sh::Shard{W,T}, cur::Vector{Vector{Int}}, j::Int,
                            nsegs::Int, kv::W) where {W,T}
    acc = zero(T)
    @inbounds for i in 1:sh.n
        sh.x[i] == zero(W) || continue
        acc += (1 - 2 * (count_ones(sh.z[i] & kv) & 1)) * sh.c[i]
    end
    @inbounds for t in 1:nsegs
        for i in sh.seg_lo[t]:(cur[t][j] - 1)
            sh.ax[i] == zero(W) || continue
            acc += (1 - 2 * (count_ones(sh.az[i] & kv) & 1)) * sh.ac[i]
        end
    end
    return acc
end
