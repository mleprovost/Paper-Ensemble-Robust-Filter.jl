export EMq_dogru

# Routine to iteratively estimate the location, scale matrix and degrees of freedom from samples
# Marzieh Hasannasab, Johannes Hertrich, Friederike Laus, and Gabriele Steidl
# Alternatives to the EM algorithm for ML estimation of location, scatter matrix, and degree of freedom
# of the Student t distribution
# Add the option to not optimize over the degree of freedom νX
function EMq_dogru(X, q::Float64, Niter::Int64=100; rtol::Float64 = 1e-5, estimate_dof = true, νX = 2.5)

    @assert q >= 0.0 && q <= 1.0

    Nx, Ne = size(X)

    # We add a Tikkonov regularization to ensure that CY is positive definite
    λ = 0.0

    #Start with sample mean and anomaly matrix of the scale matrix A_X A_X^⊤ = C_X
    # νX = 2.1
    μX = mean(X, dims = 2)[:,1]
    AX = sqrt(1/(Ne-1))*(X .- μX)

    @show μX, AX*AX', νX
    πX = Distributions.GenericMvTDist(νX, μX, PDMat(Symmetric(AX*AX' + λ*I)))

    wq1old = zeros(Ne)
    wq2old = zeros(Ne)


    sqold = zeros(Ne)
    ζqold = zeros(Ne)
    wqold = zeros(Ne)

    Wqold = zeros(Ne)

    Distributions.sqmahal!(sqold, πX, X)

    @show sqold
    for j=1:Niter
        μXold = copy(μX)
        AXold = copy(AX)
        νXold = copy(νX)

        fill!(μX, 0.0)
        fill!(AX, 0.0)

        fill!(wq1old, 0.0)
        fill!(wq2old, 0.0)

        # fill!(sqold, 0.0)
        fill!(ζqold, 0.0)
        fill!(wqold, 0.0)

        # CXold = AXold*AXold'
        # Compute Cholesky factor of C_X
        # LXold = cholesky(Symmetric(CXold + λ*I)).L

        for i=1:Ne
            xi = X[:,i]

            # sqold[i] = sum(abs2, LXold \ (xi - μXold))

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

        @show μX

        μζqold = sum(ζqold)

        for i=1:Ne
            AX[:,i] = sqrt(wqold[i]/μζqold)*(X[:,i] - μX)
        end

        @show AX*AX'

        πX = Distributions.GenericMvTDist(νXold, μX, PDMat(Symmetric(AX*AX' + λ*I)))

        # We don't need to front factors in the PDF only
        #Wq = ((1 + 1/νXold \delta_X)^(-0.5(Nx + νXold)))^(1-q)
        # Wqold = pdf(πX, X).^(1-q)
            Distributions.sqmahal!(sqold, πX, X)

        if estimate_dof == true
            factor = -0.5*(νXold + Nx)*(1-q)
            Wqold .= exp.(factor*log1p.((1/νXold)*sqold))

            #ν₋, ν₊ = bracket(x->rhsEMν(x, νXold, γXold, Nx), 1e-4, 100)
            # Hybrid solve expects an increasing function
            νX = hybridsolver(x->-rhsEMqν(x, q, wq1old, wq2old, Wqold),
                            x->-derivative_rhsEMqν(x, q, wq1old, wq2old, Wqold),
                            νXold,
                            2.001,
                            100)
        end
        # @show νX
        # , norm(μX - μXold)/(norm(μXold)*sqrt(Nx)), norm(AX*AX' - AXold*AXold')/(norm(AXold*AXold')*sqrt(Nx^2)),
        #       norm(νX - νXold)/norm(νXold)
    
        if stoppingcriterion(μX, μXold, AX*AX', AXold*AXold', νX, νXold; rtol = rtol) == false
            break
        end
        # @show j, norm(μXold - μX), norm(AXold - AX), norm(νXold - νX)
    end
    return μX, AX, νX
end
