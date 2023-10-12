
import TransportBasedInference: SeqFilter


export LikEnRFLasso

"""
$(TYPEDEF)

A structure for the likelihood-based ensemble robust filter (EnRF)

References:

$(TYPEDFIELDS)
"""

struct LikEnRFLasso<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Refresh time step"
    Δtrefresh::Float64

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Relative error for EMq procedure"
    rtol::Float64

    "Estimate dof"
    estimate_dof::Bool

    "Default νYX"
    νYX::Vector{Float64}

    "Regularization coefficient ρ in L1 penalty term in t-lasso of the form C/√{M}"
    ρ_scaling::Vector{Float64}

    "Grid search for degree of freedom"
    νYXgrid::Vector{Float64}
end

function LikEnRFLasso(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh; isfiltered = false, rtol = 1e-3, estimate_dof = true, νYX = [2.5], ρ_scaling = [1.0], νYXgrid = collect(2.5:0.5:20.0))
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"


    # if estimate_dof == false
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnRFLasso(G, ϵy, Δtdyn, Δtobs, Δtrefresh, isfiltered, rtol, estimate_dof, νYX, ρ_scaling, νYXgrid)
end

# If no filtering function is provided, use the identity in the constructor.
function LikEnRFLasso(ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh; rtol = 1e-5, estimate_dof = true, νYX = [2.5], ρ_scaling = [1.0], νYXgrid = collect(2.5:0.5:20.0))
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    # @show mod(Δtobs, Δtdyn)

    # if estimate_dof == false
    #     @show mod(Δtrefresh, Δtobs)
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnRFLasso(x -> x, ϵy, Δtdyn, Δtobs, Δtrefresh, false, rtol, estimate_dof, νYX, ρ_scaling, νYXgrid)
end



function Base.show(io::IO, enrf::LikEnRFLasso)
	println(io,"Likelihood-based Lasso EnRF  with filtered = $(enrf.isfiltered) and estimate dof = $(enrf.estimate_dof[1])")
end


function (enrf::LikEnRFLasso)(X, ystar::Array{Float64,1}, t::Float64)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    # Generate observational noise samples
    E = zeros(Ny, Ne)
    if typeof(enrf.ϵy) <: AdditiveInflation
        E .= enrf.ϵy.σ*randn(Ny, Ne) .+ enrf.ϵy.m
    elseif typeof(enrf.ϵy) <: TDistAdditiveInflation
        E .= rand(enrf.ϵy.dist, Ne)
    else
        print("Inflation type not defined")
    end

    # Estimate the location, scale and degree of freedom of the joint distribution
    X[1:Ny,:] .+= E
    # μYXiter, AYXiter, νYXiter = adaptive_EMq(X, 0.99, 100)
    # @show enrf.νX[1]

    #μYXiter, AYXiter, νYXiter, LYXiter = EMq(X, 0.98, 500; withLfactor = true, rtol = enrf.rtol, λ = 1e-8,
    #estimate_dof = enrf.estimate_dof, νX = enrf.νX[1])

    if enrf.estimate_dof == true
        μYXlasso, CYXlasso, νYXlasso  = regularized_EM_tdist_search(X, enrf.ρ_scaling[1]*1/sqrt(Ne), enrf.νYXgrid; Niter = 500,
                                                                              tol = enrf.rtol)
        enrf.νYX[1] = copy(νYXlasso)
    else
        μYXlasso, CYXlasso, νYXlasso  = regularized_EM_tdist(X, enrf.ρ_scaling[1]*1/sqrt(Ne); Niter = 500,
                                                                       tol = enrf.rtol, νX = enrf.νYX[1])   
    end

    μY = μYXlasso[1:Ny]
    CY = CYXlasso[1:Ny, 1:Ny]
    # θY = inv(CY)#θYXlasso[1:Ny,1:Ny]
    
    νY = copy(νYXlasso)

    μX = μYXlasso[Ny+1:Ny+Nx]
    # Cross scale matrix between X and Y.
    CXcrossY = CYXlasso[Ny+1:Ny+Nx,1:Ny]

    # Cholesky factorization of the precision matrix CY = LY LY^⊤
    LY = LowerTriangular(cholesky(Symmetric(CY)).L)

    # bstar = θY * (ystar - μY)
    # αstar = (νY + (ystar - μY)'*θY*(ystar - μY))/(νY+ Ny)


    zstar = LY \ (ystar - μY)
    bstar = LY' \ zstar
    αstar = (νY + sum(abs2, zstar))/(νY+ Ny)

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]
        # zi = RY \ (yi - μY)
        zi = LY \ (yi - μY)

        αi = (νY + sum(abs2, zi))/(νY+ Ny)

        # αi = (νY + (yi - μY)'*θY*(yi - μY))/(νY + Ny)
        # bi = RY' \ zi
        bi = LY' \ zi
        # bi = θY*(yi - μY)

        # @show νY, αstar, αstar/αi, sqrt(αstar/αi), norm(zstar)^2, norm(zi)^2#, norm(AX*(AY)'*bstar - AX*(AY)'*bi - (ystar - yi))
        X[Ny+1:Ny+Nx,i] =  μX + CXcrossY*bstar + sqrt(αstar/αi)*((xi - μX) - CXcrossY*bi)
    end

    # @show norm(X[Ny+1:Ny+Nx,:])

	return X
end 