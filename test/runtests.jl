using PauliOperators
using Test

@testset "Paulis.jl" begin
    include("test_operator_methods.jl")
    include("test_Pauli.jl")
    include("test_Ket.jl")
    include("test_multiplication.jl")
    include("test_addition.jl")
    include("test_allocations.jl")
    include("test_stochastic.jl")
    include("test_phase1.jl")
    include("test_truncation.jl")
    include("test_projectors.jl")
    include("test_sparse_pauli_vector.jl")
    include("test_spv_equivalence.jl")
    include("test_spv_evolution.jl")
    include("test_spv_allocations.jl")
    include("test_evolution.jl")
    include("test_analysis.jl")
    include("test_channels.jl")
    include("test_transformations.jl")
    include("test_large_N.jl")
    include("test_wide.jl")
    include("test_threaded.jl")
end

# Spawns worker processes (addprocs), so it runs outside the main testset and
# tears them down afterwards. Set PAULI_SKIP_DISTRIBUTED=1 to skip.
if get(ENV, "PAULI_SKIP_DISTRIBUTED", "0") != "1"
    include("test_distributed.jl")
end
