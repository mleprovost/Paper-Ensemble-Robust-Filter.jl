export kldiv, kldivbis

# Compute the opposite of the cross-entropy between the joint density πYX  and
# the pullback of the standard t-distribution of dof ν by the linear map SX
function kldiv(X, E, Ny::Int64, Nx::Int64, ν::Float64; withiteration::Bool=false)

    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimensions"
    @assert ν > 2 "ν must be bigger than 2"

    νXgiveny = ν + Ny
    νϵgivenx = ν + Nx

    # We add a Tikkonov regularization to ensure that the scale matrices are PD
    λ = 1e-12

    J  = -log(gamma(0.5*(νXgiveny + Nx)))
    J += log(gamma(0.5*(νXgiveny)))
    J += 0.5*Nx*log(π*νXgiveny)

    # Compute the sample mean for X
    μX = mean(view(X,Ny+1:Ny+Nx,:), dims = 2)[:,1]

    # Compute the anomaly scale for X
    AX = sqrt((ν-2)/(ν*(Ne-1)))*(X[Ny+1:Ny+Nx, :] .- μX)

    if withiteration == true
        Niter = 100

        μXiter, AXiter = locandscale(X[Ny+1:Ny+Nx,:], ν, Niter)

        μX = deepcopy(μXiter)
        AX = deepcopy(AXiter)
    end

    CX = AX*AX'
    LX = inv(cholesky(Symmetric(CX + λ*I)).L)

    # Generate the samples from Y = h(X) + ϵ
    # In this case, ϵ^i ∼ π_{ϵ | X = x^i} = St(0, α_X(x^i)C_{ϵ}, ν + Nx), with α_{X}(x) = (ν + (x - μX)^⊤ C_X^{-1} (x - μX))/(ν + Nx)

    Y = zeros(Ny, Ne)
    for i=1:Ne
        xi = X[Ny+1:Ny+Nx,i]
        hi = X[1:Ny,i]

        # Generate scaled observation samples
        zxi = LX*(xi - μX)
        αxi = (ν + zxi'*zxi)/(ν + Nx)
        ϵi = sqrt(αxi)*E[:,i]

        # Generate the samples from the likelihood
        Y[:,i] = hi + ϵi
    end

    # Aϵ = sqrt((νEgivenx-2)/(νEgivenx*(Ne-1)))*(E .- μϵ)
    # μY = mean(Y, dims = 2)[:,1]
    # AY = sqrt((ν-2)/(ν*(Ne-1)))*(Y .- μY)

    YX = [Y; X[Ny+1:Ny+Nx,:]]
    μYX = mean(YX, dims = 2)[:,1]
    AYX = sqrt((ν-2)/(ν*(Ne-1)))*(YX .- μYX)

    if withiteration == true
        Niter = 100

        μYXiter, AYXiter = locandscale(YX, ν, Niter)

        μYX = deepcopy(μYXiter)
        AYX = deepcopy(AYXiter)
    end

    # Sample covariance matrix
    # Cϵ = πϵ.Σ # Note that the notation of Distributions.jl is confusing as π.Σ is the scale matrix
    # CY = AHX*AHX' + Cϵ
    μY = μYX[1:Ny]
    μX = μYX[Ny+1:Ny+Nx]

    AY = AYX[1:Ny,:]
    AX = AYX[Ny+1:Ny+Nx,:]

    CX = AX*AX'
    CY = AY*AY'
    # Cross scale matrix
    # CXY = AX*AHX'
    CXY = AX*AY'

    # Compute the Cholesky factor of the conditional scale matrix C_{X|Y} = C_{X} - C_{X,Y} C_Y^{-1} C_{X, Y}^T
    CXgivenY = CX - CXY*inv(CY)*CXY'
    SXgivenY = inv(cholesky(Hermitian(CXgivenY + λ*I)).L)
    # @show norm(SXgivenY'*SXgivenY-inv(CXgivenY))

    LY = inv(cholesky(Symmetric(CY + λ*I)).L)
    LX = inv(cholesky(Symmetric(CX + λ*I)).L)


    for i=1:Ne
        xi = view(X, Ny+1:Ny+Nx,i)
        yi = view(Y, :,i)
        zyi = LY*(yi - μY)

        # bi  = CY\(yi - μY)
        # αi = (ν + dot(yi - μY, bi))/(ν + Ny)
        # @show norm(αi - αtdist(yi, μY, CY, ν, Ny))

        αi = (ν + zyi'*zyi)/(ν + Ny)


        Sxνi = αi^(-0.5)*(SXgivenY*(xi - μX - CXY*LY*zyi))
        J += 0.5*(νXgiveny + Nx)*(1/Ne)*log(1.0 + 1/(νXgiveny)*Sxνi'*Sxνi)
        J += 0.5*Nx*(1/Ne)*log(αi)
    end

    # Compute the sum of log of the diagonal entries of SXgivenY
    J -= logabsdet(SXgivenY)[1]

    return J
end


# Use this function if you only have samples {y^i, xi} from the joint distribution πYX
# Compute the opposite of the cross-entropy between the joint density πYX  and
# the pullback of the standard t-distribution of dof ν by the linear map SX
function kldivbis(X, Ny::Int64, Nx::Int64, ν::Float64; withiteration::Bool=false)

    Nypx, Ne = size(X)
    @assert Nypx == Ny + Nx "Wrong dimensions"
    @assert ν > 2 "ν must be bigger than 2"
    νXgiveny = ν + Ny

    # We add a Tikkonov regularization to ensure that the scale matrices are PD
    λ = 1e-12

    J  = -log(gamma(0.5*(νXgiveny + Nx)))
    J += log(gamma(0.5*(νXgiveny)))
    J += 0.5*Nx*log(π*νXgiveny)

    # Compute the sample means
    μX = mean(view(X,Ny+1:Ny+Nx,:), dims = 2)[:,1]
    μY = mean(view(X,1:Ny,:), dims = 2)[:,1]


    # Compute the anomaly scale matrices
    AX = sqrt((ν-2)/(ν*(Ne-1)))*(X[Ny+1:Ny+Nx, :] .- μX)
    AY = sqrt((ν-2)/(ν*(Ne-1)))*(X[1:Ny, :] .- μY)

    if withiteration == true
        Niter = 100

        μXiter, AXiter = locandscale(X[Ny+1:Ny+Nx,:], ν, Niter)
        μYiter, AYiter = locandscale(X[1:Ny,:], ν, Niter)

        μX = deepcopy(μXiter)
        AX = deepcopy(AXiter)

        μY = deepcopy(μYiter)
        AY = deepcopy(AYiter)
    end

    # Sample covariance matrix
    CX = AX*AX'
    CY = AY*AY'
    # Cross scale matrix
    CXY = AX*AY'

    # Compute the Cholesky factor of the conditional scale matrix C_{X|Y} = C_{X} - C_{X,Y} C_Y^{-1} C_{X, Y}^T
    CXgivenY = CX - CXY*inv(CY)*CXY'
    SXgivenY = inv(cholesky(Symmetric(CXgivenY + λ*I)).L)

    LY = inv(cholesky(Symmetric(CY + λ*I)).L)

    # @show norm(SXgivenY'*SXgivenY-inv(CXgivenY))
    # @show norm(LY'*LY-inv(CY))

    for i=1:Ne
        yi = view(X, 1:Ny, i)
        xi = view(X, Ny+1:Ny+Nx, i)
        # bi  = CY\(yi - μY)
        # αi = (ν + dot(yi - μY, bi))/(ν + Ny)

        zyi  = LY*(yi - μY)
        αi = (ν + zyi'*zyi)/(ν + Ny)

        # @show αi - (ν + dot(yi - μY, bi))/(ν + Ny)

        Sxνi = αi^(-0.5)*SXgivenY*(xi - μX - CXY*(LY'*zyi))
        J += 0.5*(νXgiveny + Nx)*(1/Ne)*log(1.0 + 1/(νXgiveny)*Sxνi'*Sxνi)
        J += 0.5*Nx*(1/Ne)*log(αi)
    end

    # Compute the sum of log of the diagonal entries of SXgivenY
    J -= logabsdet(SXgivenY)[1]

    return J
end
