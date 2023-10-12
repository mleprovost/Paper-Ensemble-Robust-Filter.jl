
import TransportBasedInference: SeqFilter


export LikEnKFLasso

"""
$(TYPEDEF)

A structure for the likelihood-based ensemble robust filter (EnRF)

References:

$(TYPEDFIELDS)
"""

struct LikEnKFLasso<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Relative error for tlasso procedure"
    rtol::Float64

    "Regularization coefficient ρ in L1 penalty term in t-lasso of the form C/√{M}"
    ρ_scaling::Vector{Float64}
end

function LikEnKFLasso(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs; isfiltered = false, rtol = 1e-3, ρ_scaling = [1.0])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    # if estimate_dof == false
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnKFLasso(G, ϵy, Δtdyn, Δtobs, isfiltered, rtol, ρ_scaling)
end

# If no filtering function is provided, use the identity in the constructor.
function LikEnKFLasso(ϵy::InflationType,
    Δtdyn, Δtobs; rtol = 1e-5, ρ_scaling = [1.0])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    # @show mod(Δtobs, Δtdyn)

    # if estimate_dof == false
    #     @show mod(Δtrefresh, Δtobs)
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnKFLasso(x -> x, ϵy, Δtdyn, Δtobs, false, rtol, ρ_scaling)
end



function Base.show(io::IO, enrf::LikEnKFLasso)
	println(io,"Likelihood-based Lasso EnKF  with filtered = $(enrf.isfiltered)")
end


function (enkf::LikEnKFLasso)(X, ystar::Array{Float64,1}, t::Float64)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enkf.ϵy) <: AdditiveInflation
        E .= enkf.ϵy.σ*randn(Ny, Ne) .+ enkf.ϵy.m
    elseif typeof(enkf.ϵy) <: TDistAdditiveInflation
        E .= rand(enkf.ϵy.dist, Ne)
    else
        print("Inflation type not defined")
    end

    # Estimate the mean and covariance of the joint distribution
    X[1:Ny,:] .+= E

    # Technically, the mean of the joint distribution does not need to be estimated 
    μYXlasso, ΣYXlasso, νYXlasso  = regularized_EM_tdist(X, enkf.ρ_scaling[1]*1/sqrt(Ne); Niter = 500, 
                                                         tol = enkf.rtol, 
                                                         νX = Inf, 
                                                         estimate_dof = false)
                                                         
    # @show norm(μYXlasso - mean(X; dims = 2)[:,1]) = 0 to machine precision                                                   
    # Regularized covarariance of the observations Y
    ΣY = ΣYXlasso[1:Ny, 1:Ny]

    # Cross scale matrix between X and Y
    ΣXcrossY = ΣYXlasso[Ny+1:Ny+Nx,1:Ny]

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]
        
        bi = ΣY \ (yi - ystar)
        X[Ny+1:Ny+Nx,i] =  xi - ΣXcrossY*bi
    end

	return X
end 
