import TransportBasedInference: SeqFilter

export SeqLiksEnKF

"""
$(TYPEDEF)

A structure for the sequential likelihood-based stochastic ensemble Kalman filter (SeqLiksEnKF)

References:

$(TYPEDFIELDS)
"""

struct SeqLiksEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Multiplicative inflation β"
    β::Float64

    "Observation matrix"
    H::Matrix{Float64}

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Index of measurement"
    idx::Array{Int64,2}
    # idx contains the dictionnary of the mapping
    # First line contains the range of integer 1:Ny
    # Second line contains the associated indice of each measurement

	"Boolean: is the covariance matrix localized"
	islocal::Bool

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function SeqLiksEnKF(G::Function, ϵy::InflationType, β,
    H::Matrix{Float64},
    Δtdyn, Δtobs, idx::Array{Int64,2}; islocal = false, isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return SeqLiksEnKF(G, ϵy, β, H, Δtdyn, Δtobs, idx, islocal, isfiltered)
end
#
# # If no filtering function is provided, use the identity in the constructor.
function SeqLiksEnKF(ϵy::InflationType, β,
    H::Matrix{Float64},
    Δtdyn, Δtobs, idx::Array{Int64,2}; islocal = false, isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return SeqLiksEnKF(x->x, ϵy, β, H, Δtdyn, Δtobs, idx, islocal, isfiltered)
end


function Base.show(io::IO, enkf::SeqLiksEnKF)
	println(io,"Sequential likelihood-based sEnKF with filtered = $(enkf.isfiltered)")
end

# In this EnKF, the inflation factor is applied to the prior samples before
# sampling from the likelihood and the covariances are computed
# using inflated prior and (not-inflated) likelihood samples.

function (enkf::SeqLiksEnKF)(X, ystar::Array{Float64,1}, t::Float64)

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    @assert size(enkf.idx, 2) == Ny

    # Assimilate sequentially the observations
    # We only use the Ny th line of the X to store the observations

    H = enkf.H

    @assert size(H) == (Ny, Nx)

    for j=1:Ny
        idx1, idx2 = enkf.idx[:,j]
        Hlocal = H[idx1:idx1,:]
        ylocal = ystar[idx1]

        μX = mean(X[Ny+1:Ny+Nx,:]; dims = 2)[:,1]

        # Apply the multiplicative inflation
        Xinfl = enkf.β*(X[Ny+1:Ny+Nx,:] .- μX) .+ μX

    #     E = Matrix([-0.14725    -0.720477   -0.352793;
    #     0.943538    0.0370835  -0.899758;
    #     0.938173   -0.162077    1.08178;
    #     1.09537     1.20312    -0.303806;
    #    -1.4555      0.115493    1.28328;
    #    -0.889658   -1.6658     -0.212225;
    #    -0.0751394   1.52574    -1.55329;
    #     0.243669    1.33002     0.604289;
    #    -0.384081    0.788309    0.377277;
    #     1.39526     0.126402   -0.548092;
    #     1.05552    -0.39294    -1.00082;
    #     0.537705    0.650342    1.18221;
    #     0.253689    0.261744    0.353331;
    #     0.203583    1.24395     1.47688;
    #     2.48703     0.659101    0.815744]')

    
        if typeof(enkf.ϵy) <: AdditiveInflation
            ysample = Hlocal*Xinfl .+ enkf.ϵy.m[idx1] + enkf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)
        elseif typeof(enkf.ϵy) <: TDistAdditiveInflation
            ysample = Hlocal*Xinfl .+ rand(enkf.ϵy.dist, Ne)[idx1:idx1,:]
        else
            print("Inflation type not defined")
        end

        # Join y,X samples
        yX = vcat(ysample, Xinfl)
        μyX = mean(yX; dims= 2)[:,1]

        AyX = 1/sqrt(Ne-1)*(yX .- μyX)

        # Estimate covariances
        Σjoint = AyX*AyX'
        ΣXy = Σjoint[2:end, 1]
        Σyy = Σjoint[1,1]

        # Localize Covariance and compute Kalman Gaussian
        K = ΣXy/Σyy

        # Sample from conditional likelihood using un-inflated samples
        if typeof(enkf.ϵy) <: AdditiveInflation
            ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ enkf.ϵy.m[idx1] + enkf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)
        elseif typeof(enkf.ϵy) <: TDistAdditiveInflation
            ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ rand(enkf.ϵy.dist, Ne)[idx1:idx1,:]
        else
            print("Inflation type not defined")
        end
        #ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ enkf.ϵy.m[idx1] + enkf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)

        for i=1:Ne
            xi = X[Ny+1:Ny+Nx,i]
            X[Ny+1:Ny+Nx,i] = xi - K*(ysample[i] - ylocal)
        end

    end

	return X
end

# function (enkf::SeqLiksEnKF)(X, ystar::Array{Float64,1}, t::Float64)
#
#     Ny = size(ystar,1)
#     Nx = size(X,1)-Ny
#     Ne = size(X, 2)
#
#     # Assimilate sequentially the observations
#     # We only use the Ny th line of the X to store the observations
#
#     for j=1:Ny
#         idx1, idx2 = enkf.idx[:,j]
#         ylocal = ystar[idx1]
#
#
#         # Sample from the likelihood
#         for i=1:Ne
#             xi = X[Ny+1:Ny+Nx,i]
#             X[Ny:Ny,i] .= enkf.F.h(xi,t)[idx2]  +
#                           enkf.ϵy.m[idx1] + dot(enkf.ϵy.σ[idx1,:], randn(Ny))
#         end
#
#         μX = mean(X[Ny+1:Ny+Nx,:], dims = 2)[:,1]
#         μY = mean(X[Ny:Ny,:], dims = 2)[:,1]
#
#         AX = 1/sqrt(Ne-1)*(X[Ny+1:Ny+Nx,:] .- μX)
#         AY = 1/sqrt(Ne-1)*(X[Ny:Ny,:] .- μY)
#
#         ΣY = AY*AY'
#
#         for i=1:Ne
#             xi = X[Ny+1:Ny+Nx,i]
#             yi = X[Ny:Ny,i]
#
#             bi = ΣY \ (yi .- ylocal)
#
#             X[Ny+1:Ny+Nx,i] = xi - AX*(AY)'*bi
#         end
#
#     end
#
# 	return X
# end
