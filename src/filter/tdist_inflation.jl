
import TransportBasedInference: InflationType

export TDistAdditiveInflation


struct TDistAdditiveInflation <: InflationType
    "Dimension of the state vector"
    Nx::Int64

    "t-distribution"
    dist::Distributions.GenericMvTDist
end


"""
        (A::TDistAdditiveInflation)(X, start::Int64, final::Int64)


Apply the additive inflation `A` to the lines `start` to `final` of an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ ∼ `A.dist`.
"""
function (A::TDistAdditiveInflation)(X, start::Int64, final::Int64)
    Ne = size(X,2)
    @assert A.Nx == final - start + 1 "final-start + 1 doesn't match the length of the additive noise"
    # @show X[start:final, 1]
    @inbounds for i=1:Ne
            col = view(X, start:final, i)
            col .+= rand(A.dist)
    end
end

"""
        (A::TDistAdditiveInflation)(X)


Apply the additive inflation `A` to an ensemble matrix `X`,
i.e. xⁱ -> xⁱ + ϵⁱ with ϵⁱ ∼ `A.dist`.
"""
(A::TDistAdditiveInflation)(X) = A(X, 1, size(X,1))

"""
        (A::TDistAdditiveInflation)(x::Array{Float64,1})

Apply the additive inflation `A` to the vector `x`,
i.e. x -> x + ϵ with ϵ ∼ `A.dist`.
"""
function (A::TDistAdditiveInflation)(x::Array{Float64,1})
    x .+= rand(A.dist)
    return x
end