"""
    variance(O::PauliSum{N}, ψ::Ket{N}) where N

Compute the variance of observable `O` in state `ψ`: `<O²> - <O>²`.
"""
function variance(O::AnyPauliSum{N}, ψ::Ket{N}) where N
    σ = KetSum(N, ComplexF64)
    for (p, ci) in O
        cj, ki = p * ψ
        curr = get(σ, ki, 0.0) + cj * ci
        σ[ki] = curr
    end

    e2 = 0.0
    for (_, v) in σ
        e2 += v' * v
    end

    e1 = get(σ, ψ, 0.0)

    return real(e2 - e1 * e1)
end

# SparsePauliVector specialization: the live buffer is x-major sorted, and
# terms sharing an x-string map ψ to the same target ket — so the KetSum
# dict of the generic method above collapses to one running sum per
# contiguous x-run. Single ordered pass, allocation-free.
function variance(v::SparsePauliVector{N,W,T}, ψ::Ket{N}) where {N,W,T}
    v.an == 0 ||
        error("variance on a SparsePauliVector with pending appends; merge first")
    kv = (ψ.v % UInt128) % W
    e2 = 0.0
    e1 = zero(ComplexF64)
    i = 1
    @inbounds while i <= v.n
        x = v.x[i]
        run = zero(ComplexF64)
        while i <= v.n && v.x[i] == x
            run += _ket_phase(v.z[i], x, kv) * v.c[i]
            i += 1
        end
        e2 += abs2(run)
        x == zero(W) && (e1 = run)
    end
    return real(e2 - e1 * e1)
end

"""
    covariance(A::PauliSum{N}, B::PauliSum{N}, ψ::Ket{N}) where N

Compute the covariance of observables `A` and `B` in state `ψ`: `<A†B> - <A†><B>`.
"""
function covariance(A::AnyPauliSum{N}, B::AnyPauliSum{N}, ψ::Ket{N}) where N
    σA = KetSum(N, ComplexF64)
    for (p, ci) in A
        cj, ki = p' * ψ
        curr = get(σA, ki, 0.0) + cj * ci'
        σA[ki] = curr
    end

    σB = KetSum(N, ComplexF64)
    for (p, ci) in B
        cj, ki = p * ψ
        curr = get(σB, ki, 0.0) + cj * ci
        σB[ki] = curr
    end

    eA = get(σA, ψ, 0.0)'
    eB = get(σB, ψ, 0.0)
    return inner_product(σA, σB) - eA * eB
end
