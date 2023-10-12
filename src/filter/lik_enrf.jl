
import TransportBasedInference: SeqFilter


export LikEnRF, LocLikEnRF

"""
$(TYPEDEF)

A structure for the likelihood-based ensemble robust filter (EnRF)

References:

$(TYPEDFIELDS)
"""

struct LikEnRF<:SeqFilter
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

    "Default νX"
    νX::Vector{Float64}
end

function LikEnRF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh; isfiltered = false, rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"


    # if estimate_dof == false
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnRF(G, ϵy, Δtdyn, Δtobs, Δtrefresh, isfiltered, rtol, estimate_dof, νX)
end

# If no filtering function is provided, use the identity in the constructor.
function LikEnRF(ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh; rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    # @show mod(Δtobs, Δtdyn)

    # if estimate_dof == false
    #     @show mod(Δtrefresh, Δtobs)
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LikEnRF(x -> x, ϵy, Δtdyn, Δtobs, Δtrefresh, false, rtol, estimate_dof, νX)
end



function Base.show(io::IO, enrf::LikEnRF)
	println(io,"Likelihood-based EnRF  with filtered = $(enrf.isfiltered) and estimate dof = $(enrf.estimate_dof[1])")
end


function (enrf::LikEnRF)(X, ystar::Array{Float64,1}, t::Float64)

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
    μYXiter, AYXiter, νYXiter, LYXiter = EMq(X, 0.98, 500; withLfactor = true, rtol = enrf.rtol, λ = 1e-8,
                                                           estimate_dof = enrf.estimate_dof, νX = enrf.νX[1])
    # They are only matching up to 
    # @show norm(AYXiter*AYXiter' - LYXiter*LYXiter')
    if enrf.estimate_dof == true
        enrf.νX[1] = copy(νYXiter)
    end

    μY = μYXiter[1:Ny]
    AY = AYXiter[1:Ny,:]
    νY = νYXiter

    μX = μYXiter[Ny+1:Ny+Nx,:]
    AX = AYXiter[Ny+1:Ny+Nx,:]


    # LY = LowerTriangular(lq(hcat(AY, sqrt(λ)*I)).L)
    # We can reuse calculations from the EMq algorithm
    LY = LowerTriangular(LYXiter[1:Ny, 1:Ny])

    zstar = LY \ (ystar - μY)
    bstar = LY' \ zstar

    αstar = (νY + sum(abs2, zstar))/(νY+ Ny)

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]
        # zi = RY \ (yi - μY)
        zi = LY \ (yi - μY)

        αi = (νY + sum(abs2, zi))/(νY + Ny)
        # bi = RY' \ zi
        bi = LY' \ zi

        # @show νY, αstar, αstar/αi, sqrt(αstar/αi), norm(zstar)^2, norm(zi)^2#, norm(AX*(AY)'*bstar - AX*(AY)'*bi - (ystar - yi))
        X[Ny+1:Ny+Nx,i] =  μX + AX*(AY)'*bstar  + sqrt(αstar/αi)*((xi - μX) - AX*(AY)'*bi)
    end

	return X
end



# Version with localization 

"""
$(TYPEDEF)

A structure for the likelihood-based ensemble robust filter (EnRF)

References:

$(TYPEDFIELDS)
"""

struct LocLikEnRF<:SeqFilter
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
    
    "Localization structure"
    Loc::Localization

    "Boolean: is state vector filtered"
    isfiltered::Bool

    "Relative error for EMq procedure"
    rtol::Float64

    "Estimate dof"
    estimate_dof::Bool

    "Default νX"
    νX::Vector{Float64}

end


function LocLikEnRF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh, Loc::Localization; isfiltered = false, rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"


    # if estimate_dof == false
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LocLikEnRF(G, ϵy, Δtdyn, Δtobs, Δtrefresh, Loc, isfiltered, rtol, estimate_dof, νX)
end

# If no filtering function is provided, use the identity in the constructor.
function LocLikEnRF(ϵy::InflationType,
    Δtdyn, Δtobs, Δtrefresh, Loc::Localization; rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    # @show mod(Δtobs, Δtdyn)

    # if estimate_dof == false
    #     @show mod(Δtrefresh, Δtobs)
    #     @assert norm(mod(Δtrefresh, Δtobs))<1e-12 "Δtrefresh should be an integer multiple of Δtobs"
    # end

    return LocLikEnRF(x -> x, ϵy, Δtdyn, Δtobs, Δtrefresh, Loc, false, rtol, estimate_dof, νX)
end



function Base.show(io::IO, enrf::LocLikEnRF)
	println(io,"Localized Likelihood-based EnRF  with filtered = $(enrf.isfiltered) and estimate dof = $(enrf.estimate_dof[1])")
end

function (enrf::LocLikEnRF)(X, ystar::Array{Float64,1}, t::Float64)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-8
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
    μYXiter, AYXiter, νYXiter = EMq(X, 0.97, 100; rtol = enrf.rtol, estimate_dof = enrf.estimate_dof, νX = enrf.νX[1])

    locXY = Locgaspari((Nx, Ny), enrf.Loc.L, enrf.Loc.Gxy)

    if enrf.estimate_dof == true
        enrf.νX[1] = copy(νYXiter)
    end

    μY = μYXiter[1:Ny]
    AY = AYXiter[1:Ny,:]
    νY = νYXiter

    μX = μYXiter[Ny+1:Ny+Nx,:]
    AX = AYXiter[Ny+1:Ny+Nx,:]


    LY = LowerTriangular(lq(hcat(AY, sqrt(λ)*I)).L)

    zstar = LY \ (ystar - μY)
    bstar = LY' \ zstar

    αstar = (νY + sum(abs2, zstar))/(νY+ Ny)

    AXY_loc = locXY .* (AX*(AY)')

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]
        # zi = RY \ (yi - μY)
        zi = LY \ (yi - μY)

        αi = (νY + sum(abs2, zi))/(νY + Ny)
        # bi = RY' \ zi
        bi = LY' \ zi

        # @show αstar/αi#, norm(AX*(AY)'*bstar - AX*(AY)'*bi - (ystar - yi))
        X[Ny+1:Ny+Nx,i] =  μX + AXY_loc*bstar  + sqrt(αstar/αi)*((xi - μX) - AXY_loc*bi)
    end

	return X
end
