export regularized_rhsEMν, loglikelihood_tdist, regularized_EM_tdist, regularized_EM_tdist_search


# rhs for root-finding, see derivation notes.
function regularized_rhsEMν(x, νold, δold::Vector{Float64}, Nx)
    Ne = length(δold)

    out = Ne*(1.0 + log(0.5*x) - digamma(0.5*x))
    
    for i=1:Ne
        # E[log τ^(i)]
        out += digamma(0.5*(νold + Nx)) - log(0.5*(νold + δold[i]))
        # - E[τ^(i)]
        out += -(νold + Nx)/(νold + δold[i])
    end
    
    return out
end

function loglikelihood_tdist(μ, C, L, ν, δ, X)
    Nx, Ne = size(X)

    # L is the cholesky factor of C, this will expedite logdeterminant calculation logdet(C) = logdet(L L^⊤) = log(det(L)^2) = 2*logdet(L) = 2*sum(log(L_{ii}))
    
    logdetC = 0.0
    @inbounds for i=1:Nx
        logdetC += log(L[i,i])
    end

    logdetC *= 2.0 

    out = Ne*log(gamma(0.5*(ν + Nx))) -Ne*log(gamma(0.5*ν)) + 0.5*Ne*ν*log(ν) - 0.5*Ne*logdetC
    out += -0.5*(ν + Nx)*sum(x-> log(ν + x), δ)    
    # for i=1:Ne
    #     out += -0.5*(ν + Nx)*log(ν + δ[i])
    # end
    
    return out
end

function regularized_EM_tdist(X, ρ; Niter::Int64=500, tol = 1e-3, νX = RobustFilter.νX_DEFAULT, 
                                    estimate_dof = false, verbose_block = false)

    Nx, Ne = size(X)
    μX = mean(X, dims = 2)[:,1]

    CX = cov(X') + ρ*I
    # θX = inv(CX)
    LXold = LowerTriangular(zeros(Nx, Nx))

    μXold = copy(μX)
    CXold = copy(CX)
    νXold = copy(νX)

    δold = zeros(Ne)
    τold = zeros(Ne)

    SτYY = zeros(Nx, Nx)

    # νX_hist = Float64[]
    # push!(νX_hist, νX)

    count = 0

    for j=1:Niter
        copy!(μXold, μX)
        copy!(CXold, CX)
        νXold = copy(νX)

        LXold .= LowerTriangular(cholesky(Symmetric(CXold)).L)

        δold .= sum.(abs2, eachcol(LXold \ (X .- μXold)))

        if νX == Inf
            fill!(τold, 1.0)
        else
            τold .= (νX + Nx)*inv.(νX .+ δold) 
        end

        # Use the weighted mean routine from StatsBase.jl, much faster and also divide the weighted sum by the sum of τold's
        μX .= mean(X, StatsBase.weights(τold), 2)[:,1]

        # Use the scattermat routine from StatsBase.jl to computed the weighted covariance matrix
        SτYY .= 1/(Ne-1)*scattermat(X, StatsBase.weights(τold); mean = μX, dims = 2)

        CX = glasso_block(SτYY, ρ; tol = tol, Niter = Niter, with_precision = false, verbose_block = false)

        if estimate_dof == true && νX != Inf
            νX = find_zero(x-> regularized_rhsEMν(x, νXold, δold, Nx), (2.0, 100.0), Bisection())
        end
        # push!(νX_hist, copy(νX))
        if νX == Inf
            # We remove the error term depending on the estimation of the degree of freedom
            if RobustFilter.stoppingcriterion(μX, μXold, CX, CXold, 1.0, 1.0; rtol = tol) == false
                break   
            end
        else
            if RobustFilter.stoppingcriterion(μX, μXold, CX, CXold, νX, νXold; rtol = tol) == false
                break   
            end
        end
        count += 1
    end

    if count == Niter
        print("EM algorithm did not converge in $(Niter)")
    end
        return μX, CX, νX#, νX_hist
end

# This version performs a parametric search over the degree of freddom to 
# estimate the mean, scale, and degree of freedom given the ensemble matrix `X` (organized by columns)
# and the regularization parameter ρ within the tlasso algorithm designed by Finegold and Drton
function regularized_EM_tdist_search(X, ρ, νXgrid; Niter::Int64=500, tol = 1e-3, verbose_block = false)
    
    Nx, Ne = size(X)
    loglikelihood_tab = zeros(length(collect(νXgrid)))

    max_loglikelihood = -Inf

    μopt = zeros(Nx)
    Copt = zeros(Nx, Nx)
    νopt = νXgrid[1]

    LXt = LowerTriangular(zeros(Nx, Nx))
    δXt = zeros(Ne)
    # idxopt = 1

    for (i, νi) in enumerate(νXgrid)
        μXt, CXt, νXt = regularized_EM_tdist(X, ρ;  tol = tol, 
                                                    νX = νi, 
                                                    estimate_dof = false, 
                                                    verbose_block = verbose_block)

        LXt .= LowerTriangular(cholesky(Symmetric(CXt)).L)                                                                          
        δXt .=  sum.(abs2, eachcol(LXt \ (X .- μXt)))
        
        # for i=1:Ne
        #     xi = X[:,i]
        #     δXt[i] = (xi - μXt)'*θXt*(xi - μXt)
        # end
        
        li = loglikelihood_tdist(μXt, CXt, LXt, νXt, δXt, X)

        loglikelihood_tab[i] = copy(li)

        if li >= max_loglikelihood
            max_loglikelihood = copy(li)
            copy!(μopt, μXt)
            copy!(Copt, CXt)
            # copy!(Lopt, LXt)
            νopt = copy(νXt)
            #δopt .= copy(δXt)
            # idxopt = copy(i)
        else
            break
        end
    end

    # @show findmax(loglikelihood_tab)[2]

    #@show νopt, idxopt
    #@show idxmin = findmax(loglikelihood_tab)[2]

    return μopt, Copt, νopt#, δopt#, idxopt
end