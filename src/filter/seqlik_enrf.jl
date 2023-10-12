
import TransportBasedInference: SeqFilter


export SeqLikEnRF

"""
$(TYPEDEF)

A structure for the sequential likelihood-based ensemble robust filter (EnRF)

References:

$(TYPEDFIELDS)
"""

struct SeqLikEnRF<:SeqFilter
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

    "Relative error for EMq procedure"
    rtol::Float64

    "Estimate dof"
    estimate_dof::Bool

    "Default νX"
    νX::Vector{Float64}

end

function SeqLikEnRF(G::Function, ϵy::InflationType, β,
    H::Matrix{Float64},
    Δtdyn, Δtobs, idx::Array{Int64,2}; 
    islocal = false, isfiltered = false, rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return SeqLikEnRF(G, ϵy, β, H, Δtdyn, Δtobs, idx, islocal, isfiltered, rtol, estimate_dof, νX)
end

# If no filtering function is provided, use the identity in the constructor.
function SeqLikEnRF(ϵy::InflationType, β,
    H::Matrix{Float64},
    Δtdyn, Δtobs, idx::Array{Int64,2}; 
    islocal = false, isfiltered = false, rtol = 1e-5, estimate_dof = true, νX = [2.5])
    @assert modfloat(Δtobs, Δtdyn) "Δtobs should be an integer multiple of Δtdyn"

    return SeqLikEnRF(x->x, ϵy, β, H, Δtdyn, Δtobs, idx, islocal, isfiltered, rtol, estimate_dof, νX)
end



function Base.show(io::IO, enrf::SeqLikEnRF)
	println(io,"Sequential likelihood-based EnRF with filtered = $(enrf.isfiltered) and estimate dof = $(enrf.estimate_dof[1])")
end


function (enrf::SeqLikEnRF)(X, ystar::Array{Float64,1}, t::Float64)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-8

    Ny = size(ystar,1)
    Nx = size(X,1)-Ny
    Ne = size(X, 2)

    @assert size(enrf.idx, 2) == Ny


    # Assimilate sequentially the observations
    # We only use the Ny th line of the X to store the observations

    H = enrf.H

    @assert size(H) == (Ny, Nx)

    for j=1:Ny
        idx1, idx2 = enrf.idx[:,j]
        Hlocal = H[idx1:idx1,:]
        ylocal = ystar[idx1]

        # Apply the multiplicative inflation
        Xinfl = deepcopy(X[Ny+1:Ny+Nx,:])#enrf.β*(X[Ny+1:Ny+Nx,:] .- μX) .+ μX

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

        if typeof(enrf.ϵy) <: AdditiveInflation
            ysample = Hlocal*Xinfl .+ enrf.ϵy.m[idx1] + enrf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)
        elseif typeof(enrf.ϵy) <: TDistAdditiveInflation
            ysample = Hlocal*Xinfl .+ rand(enrf.ϵy.dist, Ne)[idx1:idx1,:]
        else
            print("Inflation type not defined")
        end

       # Join y,X samples
        yX = vcat(ysample, Xinfl)
        # Sample mean and covariance estimates
        # μyXiter = mean(yX; dims= 2)[:,1]
        # AyXiter = 1/sqrt(Ne-1)*(yX .- μyXiter)

        μyXiter, AyXiter, νyXiter = EMq(yX, 0.97, 100; rtol = enrf.rtol, estimate_dof = enrf.estimate_dof, νX = enrf.νX[1])

        if estimate_dof == true
            enrf.νX[1] = copy(νYXiter)
        end

        # μYXiter has dimension Nx + 1
        μy = μyXiter[1]
        Ay = AyXiter[1:1,:]
        νy = νyXiter

        # @show νy

        μX = μyXiter[1+1:1+Nx,:]
        AX = AyXiter[1+1:1+Nx,:]

        Cy = (Ay*Ay')[1,1] + λ

        Ry = sqrt(Cy)

        αstar = (νy + (ylocal - μy)^2/Cy)/(νy+ 1)

        bstar = (ylocal - μy)/Cy

        # Sample from conditional likelihood using un-inflated samples
        if typeof(enrf.ϵy) <: AdditiveInflation
            ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ enrf.ϵy.m[idx1] + enrf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)
        elseif typeof(enrf.ϵy) <: TDistAdditiveInflation
            ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ rand(enrf.ϵy.dist, Ne)[idx1:idx1,:]
        else
            print("Inflation type not defined")
        end
        # ysample = Hlocal*X[Ny+1:Ny+Nx,:] .+ enrf.ϵy.m[idx1] + enrf.ϵy.σ[idx1:idx1,:]*randn(Ny, Ne)

        
        for i=1:Ne
            xi = X[Ny+1:Ny+Nx,i]
            yi = ysample[i]
            zi = (yi - μy) / Ry
            αi = (νy + zi^2)/(νy + 1)
            bi = zi / Ry
            X[Ny+1:Ny+Nx,i] =  μX + AX*(Ay)'*bstar  + sqrt(αstar/αi)*((xi - μX) - AX*(Ay)'*bi)
        end
    end
	return X
end
