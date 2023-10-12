export rhsEM2ν, derivative_rhsEM2ν, EM2,
       rhsEMqν, derivative_rhsEMqν, EMq, adaptive_EMq


# rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function rhsEM2ν(x, wq1old::Vector{Float64}, wq2old::Vector{Float64})
    Ne = length(wq1old)

    out = 0.0
    for i=1:Ne
        out += -digamma(0.5*x) + log(0.5*x) + 1.0 + wq2old[i] - wq1old[i]
    end
    return  out
end

# derivative with respect to x of the rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function derivative_rhsEM2ν(x, wq1old, wq2old)
    return -0.5*derivative_ϕtdist(0.5*x)
end


# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function EM2(X,Niter::Int64=500; estimate_dof = true, νX = νX_DEFAULT)

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 1e-10

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # νX = νX_DEFAULT
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt((νX-2)/(νX*Ne))*(X .- μX)

    wq1old = zeros(Ne)
    wq2old = zeros(Ne)
    sqold = zeros(Ne)

    for j=1:Niter
        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        fill!(AX, 0.0)

        fill!(wq1old, 0.0)
        fill!(wq2old, 0.0)

        fill!(sqold, 0.0)

        CXold = AXold*AXold'
        # # Compute Cholesky factor of C_X
        LXold = cholesky(Symmetric(CXold + λ*I)).L

        for i=1:Ne
            xi = X[:,i]

            sqold[i] = sum(abs2, LXold \ (xi - μXold))

            #############################
            ########## E step ###########
            #############################
            # E step eq (43)
            wq1old[i] = (νXold + Nx)/(νXold + sqold[i])

            # E step eq (44)
            wq2old[i] =  digamma(0.5*(νXold + Nx)) - log(0.5*(νXold + sqold[i]))

            #############################
            ########## M step ###########
            #############################

            # M step 1 eq (45) and (46)
            μX += wq1old[i]*xi
        end

        μwq1old = sum(wq1old)
        μX ./= μwq1old

        for i=1:Ne
            AX[:,i] = sqrt(wq1old[i]/Ne)*(X[:,i] - μX)
        end

        CX = AX*AX' + λ*I

        # πX = Distributions.GenericMvTDist(νXold, μX, PDMat(Symmetric(CX)))
        #
        # for i=1:Ne
        #     xi = X[:,i]
        #     Wqold[i] = pdf(πX, xi)^(1-q)
        # end

        #ν₋, ν₊ = bracket(x->rhsEMν(x, νXold, γXold, Nx), 1e-4, νX_UPPER)
        # Hybrid solve expect an increasing function
        if estimate_dof == true
            νX = hybridsolver(x->-rhsEM2ν(x, wq1old, wq2old),
                            x->-derivative_rhsEM2ν(x, wq1old, wq2old),
                            νXold,
                            νX_LOWER,
                            νX_UPPER)
        end
        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold) == false
            break
        end
        # @show j, norm(μXold - μX), norm(AXold - AX), norm(νXold - νX)
    end
    return μX, AX, νX
end

# rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function rhsEMqν(x, q::Float64, wq1old::Vector{Float64}, wq2old::Vector{Float64}, Wqold::Vector{Float64})
    Ne = length(Wqold)

    @assert q >= 0.0 && q <= 1.0
    out = 0.0
    for i=1:Ne
        out += (-ϕtdist(0.5*x) + wq2old[i] - wq1old[i] + 1.0)*Wqold[i]
    end
    return  out
end

# derivative with respect to x of the rhs for ν for ME, Algorithm 1 from
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
function derivative_rhsEMqν(x, q::Float64, wq1old, wq2old, Wqold)
    return -0.5*derivative_ϕtdist(0.5*x)*sum(Wqold)
end


# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
# Add the option to not optimize over the degree of freedom νX
function EMq(X, q::Float64, Niter::Int64=500; withLfactor::Bool = false, λ = 1e-12,  rtol::Float64 = 1e-5, estimate_dof = true, νX = νX_DEFAULT)

    @assert q >= 0.0 && q <= 1.0

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that scale matrices are positive definite
    @assert λ > 0.0 "λ should be strictly positive"

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # μX = mean(X, dims = 2)[:,1]
    μX = median(X, dims = 2)[:,1]

    AX = zeros(Nx, Ne + Nx)

    # AX[:, 1:Ne] .= sqrt((νX-2)/(νX*Ne))*(X .- μX)
    AX[:, 1:Ne] .= sqrt(1/Ne)*(X .- μX)


    # This account for the regularization of the scale matrix CX <- CX + λ I
    for i=1:Nx
        AX[i, Ne+i] = sqrt(λ)
    end


    # CX0= PDMat(Symmetric(AX*AX'))
    # CX = deepcopy(CX0)
    # idx = 0
    # while isposdef(CX) == false
    #     idx += 1
    #     CX = PDMat(Symmetric(CX0 + 10^idx*λ*I))
    # end

    # πX = Distributions.GenericMvTDist(νX, μX, CX)

    wq1old = zeros(Ne)
    wq2old = zeros(Ne)


    sqold = zeros(Ne)
    ζqold = zeros(Ne)
    wqold = zeros(Ne)

    Wqold = zeros(Ne)

    # By construction AX has always more columns than lines (because we add a block of sqrt(λ) of dimensions Nx by Nx)
    # This ensures that L is always a lower triangular matrix of dimensions Nx by Nx
    LX = LowerTriangular(lq(AX).L)

    # @show cond(LX)

    sqold .= sum.(abs2, eachcol(LX \ (X[:,1:Ne] .- μX)))
    # Distributions.sqmahal!(sqold, πX, X)

    count = 0

    νX_hist = Float64[]
    push!(νX_hist, νX)

    for j=1:Niter
        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        # fill!(AX, 0.0)

        fill!(wq1old, 0.0)
        fill!(wq2old, 0.0)

        fill!(ζqold, 0.0)
        fill!(wqold, 0.0)

        for i=1:Ne
            xi = X[:,i]

            #############################
            ########## E step ###########
            #############################
            # E step eq (43)
            wq1old[i] = (νXold + Nx)/(νXold + sqold[i])

            # E step eq (44)
            wq2old[i] =  digamma(0.5*(νXold + Nx)) - log(0.5*(νXold + sqold[i]))

            #############################
            ########## M step ###########
            #############################

            # M step 1 eq (45) and (46)
            wqold[i] = (νXold + Nx)/((νXold + sqold[i])^(1 + 0.5*(1-q)*(νXold + Nx)))

            ζqold[i] = 1/((νXold + sqold[i])^(0.5*(1-q)*(νXold + Nx)))
            μX += wqold[i]*xi
        end

        μwqold = sum(wqold)
        μX ./= μwqold

        μζqold = sum(ζqold)

        for i=1:Ne
            AX[:,i] = sqrt(wqold[i]/μζqold)*(X[:,i] - μX)
        end

        # CX0= PDMat(Symmetric(AX*AX'))
        # CX = deepcopy(CX0)
        # idx = 0
        # while isposdef(CX) == false
        #     idx += 1
        #     CX = PDMat(Symmetric(CX0 + 10^idx*λ*I))
        # end

        # πX = Distributions.GenericMvTDist(νXold, μX, CX)

        # # We don't need to front factors in the PDF only
        # #Wq = ((1 + 1/νXold \delta_X)^(-0.5(Nx + νXold)))^(1-q)
        # # Wqold = pdf(πX, X).^(1-q)
        #     Distributions.sqmahal!(sqold, πX, X)

        # By construction AX has always more columns than lines (because we add a block of sqrt(λ) of dimensions Nx by Nx)
        # This ensures that L is always a lower triangular matrix of dimensions Nx by Nx
        LX .= LowerTriangular(lq(AX).L)
        
        sqold .= sum.(abs2, eachcol(LX \ (X[:,1:Ne] .- μX)))

        if estimate_dof == true
            factor = -0.5*(νXold + Nx)*(1-q)
            Wqold .= exp.(factor*log1p.((1/νXold)*sqold))

            # @show -rhsEMqν(νX_LOWER, q, wq1old, wq2old, Wqold)
            # @show -rhsEMqν(νX_UPPER, q, wq1old, wq2old, Wqold)



            #ν₋, ν₊ = bracket(x->rhsEMν(x, νXold, γXold, Nx), 1e-4, νX_UPPER)
            # Hybrid solve expects an increasing function

            # νX = find_zero(x->-rhsEMqν(x, q, wq1old, wq2old, Wqold), (νX_LOWER, νX_UPPER), Bisection())
            νX = test_hybridsolver(x->-rhsEMqν(x, q, wq1old, wq2old, Wqold),
                            x->-derivative_rhsEMqν(x, q, wq1old, wq2old, Wqold),
                            νXold,
                            νX_LOWER,
                            νX_UPPER)
            # @show norm(νX_bisection - νX)
        end

        push!(νX_hist, νX)
    
        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold; rtol = rtol) == false
            break
        end

        count += 1
    end

    if count == Niter
        print("EMq algorithm did not converge in $(Niter)")
    end

    # We remove the block sqrt(λ)*I in AX 
    if withLfactor == true
        return μX, AX[:,1:Ne], νX, LX
    else
        return μX, AX[:,1:Ne], νX#, νX_hist
    end
end


function adaptive_EMq(X, q, Niter::Int64=100; rtol = 1e-5, estimate_dof = true, νX = νX_DEFAULT)

    νX₋ = νX_LOWER
    νX₊ = νX_UPPER
    μX, AX, νX = EMq(X, q, Niter; rtol = rtol, estimate_dof = estimate_dof, νX = νX)

    while abs(νX - νX₊)<1.0
        q *= 0.99
        μX, AX, νX = EMq(X, q, Niter; rtol = rtol, estimate_dof = estimate_dof, νX = νX)
    end

    return μX, AX, νX
end
