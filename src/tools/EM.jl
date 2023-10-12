export νX_DEFAULT, νX_LOWER, νX_UPPER,
       locandscale,
       rhsEMν, rhsaEMν, rhsECMEν,
       derivative_rhsEMν, derivative_rhsaEMν, derivative_rhsECMEν,
       EM, aEM, ECME

const νX_DEFAULT = 2.5
const νX_LOWER = 2.5
const νX_UPPER = 100.0



# Routine to iteratively estimate the location and scale matrix from samples
# Sample location and anomaly from maximum likelihood
# Modeling Non-normality Using Multivariate t: Implications for Asset Pricing, Kan and Zhou, 2016
# See also Liu and Robin, 1995
function locandscale(X, ν, Niter::Int64=100)

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-12

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt((ν-2)/(ν*(Ne-1)))*(X .- μX)

    uX = zeros(Ne)

    for j=1:Niter
        μXold = copy(μX)
        AXold = copy(AX)

        μuX = 0.0
        fill!(μX, 0.0)
        fill!(AX, 0.0)
        CXold = AXold*AXold'
        # Compute Cholesky factor of C_X^{-1}
        LXold = inv(cholesky(Symmetric(CXold + λ*I)).L)

        for i=1:Ne
            xi = X[:,i]
            # bi = (AXold*AXold')\(xi - μXold)
            # uX[i] = (ν + Nx)/(ν + dot(xi - μXold, bi))
            #inv(αtdist(xi, μX, AX*AX', ν, Nx))

            # New implementation using the Cholesky factorization of C_X^{-1}
            zi = LXold*(xi-μXold)
            uX[i] = (ν + Nx)/(ν + zi'*zi)
            # @show zi'*zi - dot((xi - μXold), (AXold*AXold')\(xi - μXold))
            μuX += uX[i]
            μX .+= uX[i]*xi
        end

        μuX *= 1/Ne
        μX .*= 1/(Ne*μuX)

        for i=1:Ne
            AX[:,i] = sqrt(uX[i]/(Ne-1))*(X[:,i] - μX)
        end
        # if norm(μXold - μX), norm(AXold - AX)
        # @show j, norm(μXold - μX), norm(AXold - AX)
    end
    return μX, AX
end


# Define some convenient routines
ϕtdist(x) = digamma(x) - log(x)

derivative_ϕtdist(x) = trigamma(x) - 1/x



# rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function rhsEMν(x, νold, γold::Vector{Float64}, Nx)
    Ne = length(γold)

    out = ϕtdist(0.5*x) - ϕtdist(0.5*(νold + Nx))
    for i=1:Ne
        γi = γold[i]
        out += 1/Ne*(γi - log(γi) - 1.0)
    end
    return  out
end

# derivative with respect to x of the rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function derivative_rhsEMν(x, νold, γold::Vector{Float64}, Nx)
    return  0.5*derivative_ϕtdist(0.5*x)
end

# rhs for ν for aME, Algorithm 2 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function rhsaEMν(x, νold, δ::Vector{Float64}, Nx)
    Ne = length(δ)
    out = ϕtdist(0.5*x) - ϕtdist(0.5*(νold + Nx))
    for i=1:Ne
        δi = δ[i]
        ratio_old = (νold + Nx)/(νold + δi)
        out += 1/Ne*(ratio_old - log(ratio_old) - 1.0)
    end
    return  out
end

# derivative with respect to x of the rhs for ν for aME, Algorithm 2 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function derivative_rhsaEMν(x, νold, δ::Vector{Float64}, Nx)
    return  0.5*derivative_ϕtdist(0.5*x)
end

# rhs for ν for ECME, Algorithm 5 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function rhsECMEν(x, δ::Vector{Float64}, Nx)
    Ne = length(δ)
    out = ϕtdist(0.5*x) - ϕtdist(0.5*(x + Nx))
    for i=1:Ne
        δi = δ[i]
        ratio = (x + Nx)/(x + δi)
        out += 1/Ne*(ratio - log(ratio) - 1.0)
    end
    return  out
end

# derivative with respect to x of the rhs for ν for ECME, Algorithm 5 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function derivative_rhsECMEν(x, δ::Vector{Float64}, Nx)
    Ne = length(δ)
    out = 0.5*derivative_ϕtdist(0.5*x) - 0.5*derivative_ϕtdist(0.5*(x + Nx))
    for i=1:Ne
        δi = δ[i]
        out += 1/Ne*((δi - Nx)/(x + δi)^2 - (δi - Nx)/((x + Nx)*(x + δi)))
    end
    return  out
end


# Stopping criterion from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function stoppingcriterion(μX, μXold, CX, CXold, νX, νXold; rtol = 1e-5)
    Nx = size(μX,1)
    loss = (norm(μX - μXold)/(norm(μXold)*sqrt(Nx)) + norm(CX - CXold)/(norm(CXold)*sqrt(Nx^2)) + 0.1*norm(νX - νXold)/norm(νXold))
    # @show loss, norm(μX - μXold)/(norm(μXold)*sqrt(Nx)), norm(CX - CXold)/(norm(CXold)*sqrt(Nx^2)), 0.1*norm(νX - νXold)/norm(νXold)
    return loss > rtol
    # maximum([norm(μX - μXold)/(norm(μXold)*sqrt(Nx));
    #                 norm(CX - CXold)/(norm(CXold)*sqrt(Nx^2));
    #                 100*norm(νX - νXold)/norm(νXold)]) > rtol
    # return sqrt(norm(μX - μXold)^2 + norm(CX - CXold)^2)/sqrt(norm(μXold)^2 + norm(CXold)^2) +
           # abs(log(νX) - log(νXold))/abs(log(νXold)) > rtol
end

# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function EM(X, Niter::Int64=100; rtol::Float64 = 1e-5, estimate_dof = true, νX = νX_DEFAULT)

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-12

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # νX = 2.1
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt((νX-2)/(νX*Ne))*(X .- μX)

    δXold = zeros(Ne)
    γXold = zeros(Ne)

    for j=1:Niter

        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        fill!(AX, 0.0)
        fill!(δXold, 0.0)
        fill!(γXold, 0.0)

        CXold = AXold*AXold'
        # # Compute Cholesky factor of C_X
        LXold = cholesky(Symmetric(CXold + λ*I)).L

        for i=1:Ne
            xi = X[:,i]

            # New implementation using the Cholesky factorization of C_X
            δXold[i] = sum(abs2, LXold \ (xi - μXold))
            γXold[i] = (νX + Nx)/(νX + δXold[i])

            μX .+= (1/Ne)*γXold[i]*xi
        end

        μγXold = mean(γXold)
        μX ./= μγXold

        for i=1:Ne
            AX[:,i] = sqrt(γXold[i]/Ne)*(X[:,i] - μX)
        end

        # Find zero
        if estimate_dof == true
            #ν₋, ν₊ = bracket(x->rhsEMν(x, νXold, γXold, Nx), 1e-4, 100)
            νX = hybridsolver(x->rhsEMν(x, νXold, γXold, Nx),
                            x->derivative_rhsEMν(x, νXold, γXold, Nx),
                            νX,
                            νX_LOWER,
                            νX_UPPER)
        end
        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold; rtol = rtol) == false
            break
        end
        # @show j, norm(μXold - μX), norm(AXold - AX), norm(νXold - νX)
    end
    return μX, AX, νX
end

# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function aEM(X, Niter::Int64=100; estimate_dof = true, νX = νX_DEFAULT)


    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-10

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # νX = 2.1
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt((νX-2)/(νX*Ne))*(X .- μX)

    δXold = zeros(Ne)
    δX = zeros(Ne)
    γXold = zeros(Ne)

    for j=1:Niter

        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        fill!(AX, 0.0)
        fill!(δXold, 0.0)
        fill!(δX, 0.0)
        fill!(γXold, 0.0)

        CXold = AXold*AXold'
        # # Compute Cholesky factor of C_X
        LXold = cholesky(Symmetric(CXold + λ*I)).L

        for i=1:Ne
            xi = X[:,i]

            # New implementation using the Cholesky factorization of C_X^{-1}
            δXold[i] = sum(abs2, LXold \ (xi - μXold))
            γXold[i] = (νX + Nx)/(νX + δXold[i])

            μX .+= (1/Ne)*γXold[i]*xi
        end

        μγXold = mean(γXold)
        μX ./= μγXold

        for i=1:Ne
            AX[:,i] = sqrt(γXold[i]/(Ne*μγXold))*(X[:,i] - μX)
        end
        
        if estimate_dof == true
            # update δX
            # # Compute Cholesky factor of C_X
            LX = cholesky(Symmetric(AX*AX' + λ*I)).L
            δX = map(i->sum(abs2, LX \ (X[:,i] - μX)), 1:Ne)

            νX = hybridsolver(x->rhsaEMν(x, νXold, δX, Nx),
                            x->derivative_rhsaEMν(x, νXold, δX, Nx),
                            νXold,
                            1e-4,
                            100)
        end

        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold) == false
            break
        end

        # @show j, norm(μXold - μX), norm(AXold - AX), norm(νXold - νX)
    end
    return μX, AX, νX
end


# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function ECME(X, Niter::Int64=100; estimate_dof = true, νX = νX_DEFAULT)

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-10

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # νX = 2.01
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt((νX-2)/(νX*Ne))*(X .- μX)

    δXold = zeros(Ne)
    δX = zeros(Ne)
    γXold = zeros(Ne)

    for j=1:Niter

        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        fill!(AX, 0.0)
        fill!(δXold, 0.0)
        fill!(δX, 0.0)
        fill!(γXold, 0.0)

        CXold = AXold*AXold'
        # # Compute Cholesky factor of C_X
        LXold = cholesky(Symmetric(CXold + λ*I)).L

        for i=1:Ne
            xi = X[:,i]

            # New implementation using the Cholesky factorization of C_X
            δXold[i] = sum(abs2, LXold \ (xi - μXold))
            γXold[i] = (νX + Nx)/(νX + δXold[i])

            μX .+= (1/Ne)*γXold[i]*xi
        end

        μγXold = mean(γXold)
        μX ./= μγXold

        for i=1:Ne
            AX[:,i] = sqrt(γXold[i]/(Ne*μγXold))*(X[:,i] - μX)
        end

        if estimate_dof == true
            # update δX
            # # Compute Cholesky factor of C_X
            LX = cholesky(Symmetric(AX*AX' + λ*I)).L

            δX = map(i->sum(abs2, LX \ (X[:,i] - μX)), 1:Ne)

            νX = hybridsolver(x->rhsECMEν(x, δX, Nx),
                            x->derivative_rhsECMEν(x, δX, Nx),
                            νXold,
                            2.00001,
                            1000)
        end

        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold) == false
            break
        end

        # @show j, norm(μXold - μX), norm(AXold - AX), norm(νXold - νX)
    end
    return μX, AX, νX
end
