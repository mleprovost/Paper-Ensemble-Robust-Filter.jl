

export LiksEnKF, LocLiksEnKF

"""
$(TYPEDEF)

A structure for the likelihood-based stochastic ensemble Kalman filter (LiksEnKF)

References:

$(TYPEDFIELDS)
"""

struct LiksEnKF<:SeqFilter
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
end

function LiksEnKF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs; isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LiksEnKF(G, ϵy, Δtdyn, Δtobs, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LiksEnKF(ϵy::InflationType, Δtdyn, Δtobs)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LiksEnKF(x -> x, ϵy, Δtdyn, Δtobs, false)
end



function Base.show(io::IO, enkf::LiksEnKF)
	println(io,"Likelihood-based sEnKF with filtered = $(enkf.isfiltered)")
end


function (enkf::LiksEnKF)(X, ystar::Array{Float64,1}, t::Float64)

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
    
   #  E = Matrix([-0.14725    -0.720477   -0.352793;
   #  0.943538    0.0370835  -0.899758;
   #  0.938173   -0.162077    1.08178;
   #  1.09537     1.20312    -0.303806;
   # -1.4555      0.115493    1.28328;
   # -0.889658   -1.6658     -0.212225;
   # -0.0751394   1.52574    -1.55329;
   #  0.243669    1.33002     0.604289;
   # -0.384081    0.788309    0.377277;
   #  1.39526     0.126402   -0.548092;
   #  1.05552    -0.39294    -1.00082;
   #  0.537705    0.650342    1.18221;
   #  0.253689    0.261744    0.353331;
   #  0.203583    1.24395     1.47688;
   #  2.48703     0.659101    0.815744]')
    # Add observational noise samples as we can only sample from the likelihood model
    X[1:Ny,:] .+= E

    μX = mean(X[Ny+1:Ny+Nx,:], dims = 2)[:,1]
    μY = mean(X[1:Ny,:], dims = 2)[:,1]

    AX = 1/sqrt(Ne-1)*(X[Ny+1:Ny+Nx,:] .- μX)
    AY = 1/sqrt(Ne-1)*(X[1:Ny,:] .- μY)

    ΣY = AY*AY'
    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]

        bi = ΣY \ (yi - ystar)

        X[Ny+1:Ny+Nx,i] = xi - AX*(AY)'*bi
    end

	return X
end



"""
$(TYPEDEF)

A structure for the likelihood-based stochastic ensemble Kalman filter (LiksEnKF)

References:

$(TYPEDFIELDS)
"""

struct LocLiksEnKF<:SeqFilter
    "Filter function"
    G::Function

    "Standard deviations of the measurement noise distribution"
    ϵy::InflationType

    "Time step dynamic"
    Δtdyn::Float64

    "Time step observation"
    Δtobs::Float64

    "Localization structure"
    Loc::Localization

    "Boolean: is state vector filtered"
    isfiltered::Bool
end

function LocLiksEnKF(G::Function, ϵy::InflationType,
    Δtdyn, Δtobs, Loc::Localization; isfiltered = false)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocLiksEnKF(G, ϵy, Δtdyn, Δtobs, Loc, isfiltered)
end

# If no filtering function is provided, use the identity in the constructor.
function LocLiksEnKF(ϵy::InflationType, Δtdyn, Δtobs, Loc::Localization)
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return LocLiksEnKF(x -> x, ϵy, Δtdyn, Δtobs, Loc, false)
end


function Base.show(io::IO, enkf::LocLiksEnKF)
	println(io,"Localized Likelihood-based sEnKF with filtered = $(enkf.isfiltered)")
end


# Version with localization 

function (enkf::LocLiksEnKF)(X, ystar::Array{Float64,1}, t::Float64)

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


    locXY = Locgaspari((Nx, Ny), enkf.Loc.L, enkf.Loc.Gxy)
    
   #  E = Matrix([-0.14725    -0.720477   -0.352793;
   #  0.943538    0.0370835  -0.899758;
   #  0.938173   -0.162077    1.08178;
   #  1.09537     1.20312    -0.303806;
   # -1.4555      0.115493    1.28328;
   # -0.889658   -1.6658     -0.212225;
   # -0.0751394   1.52574    -1.55329;
   #  0.243669    1.33002     0.604289;
   # -0.384081    0.788309    0.377277;
   #  1.39526     0.126402   -0.548092;
   #  1.05552    -0.39294    -1.00082;
   #  0.537705    0.650342    1.18221;
   #  0.253689    0.261744    0.353331;
   #  0.203583    1.24395     1.47688;
   #  2.48703     0.659101    0.815744]')
    # Add observational noise samples as we can only sample from the likelihood model
    X[1:Ny,:] .+= E

    μX = mean(X[Ny+1:Ny+Nx,:], dims = 2)[:,1]
    μY = mean(X[1:Ny,:], dims = 2)[:,1]

    AX = 1/sqrt(Ne-1)*(X[Ny+1:Ny+Nx,:] .- μX)
    AY = 1/sqrt(Ne-1)*(X[1:Ny,:] .- μY)

    ΣY = AY*AY'

    AXY_loc = locXY .* (AX*(AY)')

    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        yi = X[1:Ny,i]

        bi = ΣY \ (yi - ystar)
       
        X[Ny+1:Ny+Nx,i] = xi - AXY_loc*bi
    end

	return X
end
