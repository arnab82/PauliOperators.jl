using PauliOperators
using Test
using Random

# Multithreaded single-operator rotation. The contract is simple and strict:
# evolve_threaded! must produce EXACTLY what evolve! produces, on both
# backends and at every storage width. These tests are meaningful only when
# Julia is started with --threads>1; with one thread they still run and
# exercise the serial fallback.

# Build an operator with enough terms to cross THREADED_ROTATE_MIN, so the
# threaded branch is actually taken rather than the fallback.
function _big_operator(N, nterms; seed = 11)
    Random.seed!(seed)
    O = PauliSum(N, ComplexF64)
    while length(O) < nterms
        O[rand(PauliBasis{N})] = (2 * rand() - 1) + (2 * rand() - 1) * im
    end
    return O
end

_maxdiff(a, b) = begin
    ks = union(keys(a), keys(b))
    isempty(ks) ? 0.0 : maximum(abs(get(a, k, 0.0 + 0im) - get(b, k, 0.0 + 0im)) for k in ks)
end

@testset "threaded rotation" begin

    @info "threaded tests running with $(Threads.nthreads()) thread(s)"

    @testset "chunk ranges partition exactly" begin
        for (n, k) in ((10, 3), (10, 1), (3, 5), (4096, 8), (0, 4))
            rs = PauliOperators._chunk_ranges(n, k)
            @test length(rs) == k
            covered = vcat(collect.(rs)...)
            @test sort(covered) == collect(1:n)     # a partition: no gaps, no overlap
        end
    end

    @testset "falls back to serial below the threshold" begin
        N = 20
        O = _big_operator(N, 16)
        @test length(O) < THREADED_ROTATE_MIN
        G = PauliBasis(Pauli(N; X = [1, 2]))
        a = deepcopy(O); evolve!(a, G, 0.3)
        b = deepcopy(O); evolve_threaded!(b, G, 0.3)
        @test _maxdiff(a, b) < 1e-14
    end

    # The core guarantee, swept across the storage-word boundary.
    @testset "threaded == serial, single rotation (N=$N, $backend)" for
            N in (20, 200), backend in (:dict, :spv)
        O0 = _big_operator(N, 6000)          # > THREADED_ROTATE_MIN
        G = PauliBasis(Pauli(N; X = [1, 2]))

        ref = deepcopy(O0)
        evolve!(ref, G, 0.37)

        if backend == :dict
            got = deepcopy(O0)
            evolve_threaded!(got, G, 0.37)
            @test _maxdiff(ref, got) < 1e-12
            @test length(got) == length(ref)
        else
            v = SparsePauliVector(O0)
            evolve_threaded!(v, G, 0.37)
            @test PauliOperators.check_spv(v)
            @test length(v) == length(ref)
            @test _maxdiff(ref, PauliSum(v)) < 1e-12
        end
    end

    @testset "threaded == serial, rotation sequence (N=$N, $backend)" for
            N in (20, 200), backend in (:dict, :spv)
        O0 = _big_operator(N, 6000)
        W = word_type(N)
        gens = PauliBasis{N,W}[PauliBasis(Pauli(N; X = [k, k + 1])) for k in 1:5]
        append!(gens, PauliBasis{N,W}[PauliBasis(Pauli(N; Y = [k])) for k in 1:5])
        angs = [0.05k for k in 1:length(gens)]

        ref = deepcopy(O0)
        for (G, θ) in zip(gens, angs)
            evolve!(ref, G, θ)
            truncate!(ref, CoeffTruncation(1e-9))
        end

        got = backend == :dict ? deepcopy(O0) : SparsePauliVector(O0)
        evolve_threaded!(got, gens, angs;
                         truncation = CoeffTruncation(1e-9))
        gotd = backend == :dict ? got : PauliSum(got)
        backend == :spv && @test PauliOperators.check_spv(got)
        @test length(gotd) == length(ref)
        @test _maxdiff(ref, gotd) < 1e-12
    end

    @testset "threaded=false matches threaded=true" begin
        N = 200
        O0 = _big_operator(N, 6000)
        G = PauliBasis(Pauli(N; Z = [3, 190]))    # touches a high qubit
        v_on  = SparsePauliVector(O0); evolve_threaded!(v_on,  G, 0.21; threaded = true)
        v_off = SparsePauliVector(O0); evolve_threaded!(v_off, G, 0.21; threaded = false)
        @test _maxdiff(PauliSum(v_on), PauliSum(v_off)) < 1e-14
    end

    @testset "reduced-precision coefficients survive the threaded path" begin
        N = 200
        O32 = PauliSum(N, ComplexF32)
        for (p, c) in _big_operator(N, 6000)
            O32[p] = ComplexF32(c)
        end
        v = SparsePauliVector(O32)
        @test eltype(v.c) === ComplexF32
        evolve_threaded!(v, PauliBasis(Pauli(N; X = [1, 2])), 0.3)
        @test PauliOperators.check_spv(v)
        @test eltype(v.c) === ComplexF32
    end

    @testset "sequence length is validated" begin
        N = 20
        v = SparsePauliVector(_big_operator(N, 32))
        W = word_type(N)
        @test_throws DimensionMismatch evolve_threaded!(
            v, PauliBasis{N,W}[PauliBasis(Pauli(N; X = [1, 2]))], [0.1, 0.2])
    end
end
