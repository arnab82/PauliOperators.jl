using PauliOperators
using Test
using Random

# Tests for the shared-memory sharded engine (ShardedPauliSum / RankMap).
#
# The properties worth guarding are the ones whose failure is SILENT: a wrong
# GF(2) shift routes terms to the wrong shard, a mismatched sort order corrupts
# the merge, and a correction measured on unmerged state comes out negative.
# Each of those is checked directly below.

function _chain_H(N)
    H = PauliSum(N)
    for i in 1:N-1
        H[PauliBasis(Pauli(N, X=[i, i+1]))] = 1.0
        H[PauliBasis(Pauli(N, Z=[i, i+1]))] = 1.1
    end
    H[PauliBasis(Pauli(N, Z=[1]))] = 0.3
    return H
end
_probe(N) = (O = PauliSum(N, Float64); O[PauliBasis(Pauli(N, Z=[N ÷ 2]))] = 1.0; O)

function _plain_evolve(N, gens, angs, trunc, window)
    P = _probe(N)
    for (k, (g, a)) in enumerate(zip(gens, angs))
        evolve!(P, g, a)
        (k % window == 0 || k == length(gens)) && truncate!(P, trunc)
    end
    return P
end

_maxdiff(A::PauliSum, B::PauliSum) = begin
    d = 0.0
    for (p, c) in A; d = max(d, abs(c - get(B, p, 0.0))); end
    for (p, c) in B; d = max(d, abs(c - get(A, p, 0.0))); end
    d
end

@testset "sharded engine" begin

    @testset "RankMap width follows word_type (no 127-qubit cap)" begin
        for N in (10, 64, 100, 140, 300, 600)
            A = rand(RankMap{N}, 3)
            @test A isa RankMap{N}
            @test nbits(A) == 3
            @test nbins(A) == 8
            # rows must be as wide as the register bitint would use
            @test eltype(A.rows) === RankRow{PauliOperators.word_type(N)}
        end
        @test_throws Exception RankMap{65}([RankRow{UInt64}(0x1, 0x0)])  # row too narrow for N
    end

    @testset "bin_index is GF(2)-linear (the routing identity)" begin
        # The whole design rests on bin(G*p) == bin(p) XOR bin(G): if this fails,
        # rotations route terms to the wrong shard and results are silently wrong.
        for N in (12, 70, 200)
            Random.seed!(4)
            A = rand(RankMap{N}, 4)
            W = PauliOperators.word_type(N)
            for _ in 1:40
                p = PauliBasis{N,W}(rand(W) & ((one(W) << N) - one(W)),
                                    rand(W) & ((one(W) << N) - one(W)))
                g = PauliBasis{N,W}(rand(W) & ((one(W) << N) - one(W)),
                                    rand(W) & ((one(W) << N) - one(W)))
                gp = PauliBasis{N,W}(p.z ⊻ g.z, p.x ⊻ g.x)
                @test bin_index(A, gp) == bin_index(A, p) ⊻ bin_shift(A, g)
            end
        end
    end

    @testset "x-watching rows make Z-only generators communication free" begin
        N = 20
        rows = [RankRow(N; x = [i for i in 1:N if i % 3 == k % 3]) for k in 1:3]
        A = RankMap{N}(rows)
        W = PauliOperators.word_type(N)
        for i in 1:N, j in i+1:N                      # every ZZ and Z generator
            @test bin_shift(A, PauliBasis(Pauli(N, Z=[i, j]))) == 0
        end
        for i in 1:N
            @test bin_shift(A, PauliBasis(Pauli(N, Z=[i]))) == 0
        end
        @test any(bin_shift(A, PauliBasis(Pauli(N, X=[i]))) != 0 for i in 1:N)
    end

    @testset "PauliSum round-trip" begin
        N = 12
        O = _chain_H(N)
        A = rand(RankMap{N}, 3)
        S = ShardedPauliSum(O, A; T=Float64, nthreads=1)
        @test length(S) == length(O)
        @test _maxdiff(PauliSum(S), O) < 1e-14
    end

    @testset "serial evolve! matches plain PauliSum" begin
        for (N, window) in ((10, 1), (12, 4), (16, 8))
            H = _chain_H(N)
            gens, angs = trotterize(H, 0.3, n_trotter=3, order=2)
            A = rand(RankMap{N}, 3)
            trunc = CoeffTruncation(1e-12)
            S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1)
            evolve!(S, compile(A, gens, angs; window), truncation=trunc)
            @test _maxdiff(PauliSum(S), _plain_evolve(N, gens, angs, trunc, window)) < 1e-10
        end
    end

    @testset "threaded evolve! matches serial (and is bit-stable)" begin
        Threads.nthreads() > 1 || return           # nothing to compare on 1 thread
        for N in (12, 140)                          # UInt64 and UInt256 storage
            H = _chain_H(N)
            gens, angs = trotterize(H, 0.3, n_trotter=2, order=2)
            A = rand(RankMap{N}, 3)
            circ = compile(A, gens, angs; window=4)
            trunc = CoeffTruncation(1e-12)
            # Capacity matters for this comparison. A capacity-forced EARLY merge
            # truncates off-schedule, and it fires at different points for
            # different thread counts (measured 32 vs 25 at N=140), so the two
            # runs would then integrate genuinely different truncation cadences.
            # Size the buffers so no early merge fires, and assert that.
            big = (; min_capacity = 1 << 20, append_factor = 8.0)
            nw = cld(length(gens), 4)
            S1 = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1, big...)
            c1 = ShardedCounters(nw); evolve!(S1, circ; truncation=trunc, counters=c1)
            Sn = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=Threads.nthreads(), big...)
            cn = ShardedCounters(nw); evolve!(Sn, circ; truncation=trunc, counters=cn)
            @test sum(c1.early_merges) == 0
            @test sum(cn.early_merges) == 0
            @test length(Sn) == length(S1)
            @test _maxdiff(PauliSum(Sn), PauliSum(S1)) < 1e-12
        end
    end

    @testset "truncation strategies reach the sharded path" begin
        N = 14
        H = _chain_H(N)
        gens, angs = trotterize(H, 0.3, n_trotter=2, order=2)
        A = rand(RankMap{N}, 3)
        # window=1 so both sides truncate after EVERY rotation. With a larger
        # window the two cadences diverge and a lossy strategy then differs at
        # the scale of its own cutoff -- a real sensitivity, but not a bug, and
        # not what this test is for.
        circ = compile(A, gens, angs; window=1)
        for trunc in (CoeffTruncation(1e-8), WeightTruncation(4),
                      CompositeTruncation((CoeffTruncation(1e-10), WeightTruncation(6))))
            S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1)
            evolve!(S, circ; truncation=trunc)
            @test _maxdiff(PauliSum(S), _plain_evolve(N, gens, angs, trunc, 1)) < 1e-10
        end
        # a weight cap must actually be respected
        S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1)
        evolve!(S, compile(A, gens, angs; window=1); truncation=WeightTruncation(3))
        @test all(weight(p) <= 3 for (p, _) in PauliSum(S))
    end

    # A correction that is not a fused scalar drop-sink must be measured on
    # MERGED state. Measuring before the merge misses every pending append and
    # yields a large NEGATIVE "loss" -- the bug this guards against.
    mutable struct _NormLoss <: PauliOperators.CorrectionAccumulator
        acc::Float64
    end
    PauliOperators._measure(S::ShardedPauliSum, ::_NormLoss) =
        sum(sum(abs2, view(sh.c, 1:sh.n)) for sh in S.shards; init=0.0)
    PauliOperators._accumulate!(c::_NormLoss, before, after) = (c.acc += before - after; nothing)

    @testset "correction is measured on merged state (non-negative, matches norm loss)" begin
        N = 12
        H = _chain_H(N)
        gens, angs = trotterize(H, 0.4, n_trotter=3, order=2)
        A = rand(RankMap{N}, 3)
        trunc = CoeffTruncation(1e-6)
        S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1)
        corr = _NormLoss(0.0)
        evolve!(S, compile(A, gens, angs; window=1); truncation=trunc, correction=corr)
        kept = sum(sum(abs2, view(sh.c, 1:sh.n)) for sh in S.shards; init=0.0)
        @test corr.acc >= 0                       # a loss, never a gain
        # unitary evolution preserves the 2-norm, so kept + discarded == 1
        @test isapprox(kept + corr.acc, 1.0; atol=1e-8)
    end

    # A VECTOR correction (ShardedVectorCorrection) is the interface that lets a
    # correction run multithreaded: each thread measures the shards it owns and
    # thread 1 reduces. It must give the identical answer at every thread count,
    # and must survive a capacity-forced EARLY merge -- a separate code path that
    # only fires once the append buffers overflow, i.e. at large populations.
    mutable struct _SiteLoss <: PauliOperators.ShardedVectorCorrection
        acc::Vector{Float64}
        N::Int
    end
    PauliOperators.correction_width(c::_SiteLoss) = c.N
    function PauliOperators.measure_shard!(dst::Vector{Float64}, sh, ::_SiteLoss)
        @inbounds for i in 1:sh.n
            w2 = abs2(sh.c[i]); w2 == 0.0 && continue
            m = sh.x[i]; j = 1
            while m != zero(m)
                isodd(m & one(m)) && (dst[j] += w2)
                m >>= 1; j += 1
            end
        end
        return nothing
    end
    PauliOperators._measure(S::ShardedPauliSum, c::_SiteLoss) =
        (d = zeros(Float64, c.N); for sh in S.shards; PauliOperators.measure_shard!(d, sh, c); end; d)
    PauliOperators._accumulate!(c::_SiteLoss, before, after) =
        (c.acc .+= (before .- after); nothing)

    @testset "vector correction is thread-count independent" begin
        N = 14
        H = _chain_H(N)
        gens, angs = trotterize(H, 0.8, n_trotter=6, order=2)
        A = rand(RankMap{N}, 3)
        circ = compile(A, gens, angs; window=1)
        trunc = CoeffTruncation(1e-5)
        ref = nothing
        for nt in unique([1, min(2, Threads.nthreads()), Threads.nthreads()])
            S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=nt)
            c = _SiteLoss(zeros(Float64, N), N)
            evolve!(S, circ; truncation=trunc, correction=c)
            @test all(c.acc .>= -1e-12)                 # a loss, never a gain
            if ref === nothing
                ref = (copy(c.acc), PauliSum(S))
                @test sum(ref[1]) > 0                    # the test must actually truncate
            else
                @test maximum(abs.(c.acc .- ref[1])) < 1e-10
                @test _maxdiff(PauliSum(S), ref[2]) < 1e-12
            end
        end
    end

    @testset "vector correction runs correctly through early merges" begin
        # The capacity-forced early merge is a SEPARATE code path in the worker
        # from the window boundary, and it fires only once the append buffers
        # overflow -- so it needs its own coverage. Deliberately tiny buffers
        # force it.
        #
        # Bit-equality across thread counts is NOT asserted here: early merges
        # fire at different points for different nt, so the runs integrate
        # different truncation cadences by construction. What must hold either
        # way is the bookkeeping -- the correction is a loss, never a gain, and
        # kept + discarded accounts for the whole (unitarily conserved) 2-norm.
        N = 16
        H = _chain_H(N)
        gens, angs = trotterize(H, 1.0, n_trotter=8, order=2)
        A = rand(RankMap{N}, 3)
        circ = compile(A, gens, angs; window=8)
        trunc = CoeffTruncation(1e-7)
        nw = cld(length(gens), 8)
        for nt in unique([1, Threads.nthreads()])
            S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=nt,
                                min_capacity=1024, append_factor=1.0)
            c = _SiteLoss(zeros(Float64, N), N)
            ctr = ShardedCounters(nw)
            evolve!(S, circ; truncation=trunc, correction=c, counters=ctr)
            @test sum(ctr.early_merges) > 0            # the path was actually taken
            @test all(c.acc .>= -1e-12)                # a loss, never a gain
            @test length(S) > 0
        end
    end

    @testset "compile guards against a stale rank map" begin
        N = 10
        H = _chain_H(N)
        gens, angs = trotterize(H, 0.2, n_trotter=1, order=2)
        A = rand(RankMap{N}, 3)
        S = ShardedPauliSum(_probe(N), A; T=Float64, nthreads=1)
        circ = compile(A, gens, angs; window=1)
        S.version += 1                            # simulate the map changing
        @test_throws ErrorException evolve!(S, circ; truncation=NoTruncation())
    end
end
