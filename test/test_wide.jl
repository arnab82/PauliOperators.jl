using PauliOperators
using Test

# Reduced-precision coefficients at wide N.
#
# Width selection, wide-N algebra, Dict/SPV evolve and truncate parity past the
# word boundary are all covered by test_large_N.jl; this file covers only what
# that one does not: carrying Float32/ComplexF32 coefficients, which halves
# coefficient bandwidth and is the difference between fitting and not fitting a
# 10x10x10 = 1000-qubit lattice in memory.

@testset "reduced-precision coefficients at wide N" begin

    @testset "Dict PauliSum preserves the coefficient type (N=$N)" for N in (200, 1000)
        G = PauliBasis(Pauli(N; X = [2, 3]))
        O32 = PauliSum(N, ComplexF32)
        O32[PauliBasis(Pauli(N; Z = [1, 2]))] = ComplexF32(1)
        b32 = evolve(O32, G, 0.3)
        @test valtype(b32) === ComplexF32
        @test length(b32) == 2                       # anticommuting generator branches

        O64 = PauliSum(N, ComplexF64)
        O64[PauliBasis(Pauli(N; Z = [1, 2]))] = 1.0 + 0.0im
        b64 = evolve(O64, G, 0.3)
        @test isapprox(sum(abs2, values(b32)), sum(abs2, values(b64)); atol = 1e-5)
        @test isapprox(sum(abs2, values(b64)), 1.0; atol = 1e-12)   # cos^2 + sin^2
    end

    # The SPV kernels take the coefficient magnitude straight from the stored
    # coefficient, so a Float32 container hands `should_drop` a Float32. That
    # signature used to demand Float64, making ComplexF32 unusable on the SPV
    # path (MethodError) while the Dict path accepted it.
    @testset "SparsePauliVector accepts ComplexF32 (N=$N)" for N in (200, 1000)
        G = PauliBasis(Pauli(N; X = [2, 3]))
        O32 = PauliSum(N, ComplexF32)
        O32[PauliBasis(Pauli(N; Z = [1, 2]))] = ComplexF32(1)

        v32 = SparsePauliVector(O32)
        @test v32 isa SparsePauliVector{N,word_type(N),ComplexF32}
        evolve!(v32, G, 0.3)
        truncate!(v32, CoeffTruncation(1e-4))
        @test PauliOperators.check_spv(v32)
        @test eltype(v32.c) === ComplexF32

        O64 = PauliSum(N, ComplexF64)
        O64[PauliBasis(Pauli(N; Z = [1, 2]))] = 1.0 + 0.0im
        v64 = SparsePauliVector(O64)
        evolve!(v64, G, 0.3)
        truncate!(v64, CoeffTruncation(1e-4))

        @test length(v32) == length(v64)
        @test isapprox(sum(abs2, v32.c[1:v32.n]), sum(abs2, v64.c[1:v64.n]); atol = 1e-5)
        # the point of the exercise: half the coefficient bandwidth
        @test sizeof(ComplexF32) * 2 == sizeof(ComplexF64) * 1
    end

    @testset "weight-based truncation also works on Float32 coefficients" begin
        N = 200
        O32 = PauliSum(N, ComplexF32)
        O32[PauliBasis(Pauli(N; Z = [1]))]          = ComplexF32(1)
        O32[PauliBasis(Pauli(N; Z = [1, 2, 3, 4]))] = ComplexF32(0.5)
        v32 = SparsePauliVector(O32)
        truncate!(v32, WeightTruncation(2))
        @test length(v32) == 1
        @test PauliOperators.check_spv(v32)
    end
end
