export glasso_block, glasso

# Version based on Algorithm 2 Witten et al. 2011
function glasso_block(S, ρ; tol = 1e-3, Niter = 500, with_precision = false, verbose_block = false)
    
    Nx = size(S,1)
    
    W = zero(S)
    
    if Nx == 1
        if with_precision == true
            θ = [1/W[1,1]]
            return W, θ
        else
            return W
        end
    end

    if with_precision == true
        θ = zero(S)
    end
    
    # Build indicator matrix 1_{|S[i, i′]| > ρ}
    indicator = abs.(S) .> ρ
    # Add this step to remove diagonal entries, as we use an undirected graph. 
    # Self-loops are not defined
    indicator[diagind(indicator)] .= false
    
    # Build associated adjacency matrix adj
    adj = convert(SparseMatrixCSC{Int64,Int64}, indicator)
    
    # Identify connected blocks. We use an undirected graph from Graphs.jl
    blocks = connected_components(SimpleGraph(adj))

    if verbose_block == true
        @show blocks
    end
    
    for (j, block) in enumerate(blocks)
#         @show block
        if with_precision == true
            W[block, block], θ[block, block] = glasso(S[block,block], ρ; tol = tol, Niter = Niter, with_precision = true)
        else
            W[block, block] = glasso(S[block,block], ρ; tol = tol, Niter = Niter, with_precision = false)
        end            
    end
    
    if with_precision == true
        return W, θ
    else
        return W
    end
end


function glasso(S, ρ; tol = 1e-3, Niter = 500, with_precision = false)
    
    Nx = size(S, 1)

    W = S + ρ*I
    
    if Nx == 1
        if with_precision == true
            θ = [1/W[1,1]]
            return W, θ
        else
            return W
        end
    end
    
    W11 = zeros(Nx-1, Nx-1)
    b = zeros(Nx-1)
    βj = zeros(Nx-1)
    
    Wold = copy(W)
    
    jminus = zeros(Int64, Nx-1)
    
    # Iterate until convergence
    for i=1:Niter
        for j = 1:Nx
            jminus .= setdiff(1:Nx, j)
            
            # Compute a square-root factorization of W11
            W11 .= W[jminus, jminus]
            
            W11sqrt = sqrt(W11) #W_11^(1/2)
            b .= W11sqrt \ S[jminus,j] # W_11^(-1/2) * s_12
            
            # We need this rescaling by ρ/(Nx-1 due to the definition of the function fit in Lasso.jl, 
            # that includes a factor 1/(Nx-1) upfront of the log-likelihood term
            # fit(LassoPath, X, y, d=Normal(), l=canonicallink(d); ...)
#             fits a linear or generalized linear Lasso path given the design
#             matrix `X` and response `y`:

#             ``\underset{\beta,b_0}{\operatorname{argmin}} -\frac{1}{N} \mathcal{L}(y|X,\beta,b_0) + \lambda\left[(1-\alpha)\frac{1}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1\right]``

#             where ``0 \le \alpha \le 1`` sets the balance between ridge (``\alpha = 0``)
#             and lasso (``\alpha = 1``) regression, and ``N`` is the number of rows of ``X``.
#             The optional argument `d` specifies the conditional distribution of
#             response, while `l` specifies the link function. Lasso.jl inherits
#             supported distributions and link functions from GLM.jl. The default
#             is to fit an linear Lasso path, i.e., `d=Normal(), l=IdentityLink()`,
#             or ``\mathcal{L}(y|X,\beta) = -\frac{1}{2}\|y - X\beta - b_0\|_2^2 + C``
            # Nx-1 is the number of rows for W11sqrt
            βj .= fit(LassoPath, W11sqrt, b, λ=[ρ/(Nx-1)],
                         standardize=false, intercept=false, cd_tol = 1e-9).coefs
            
            # Update W using that βj is a sparse vector
            W[j, jminus] = W11*sparse(βj)
            W[jminus, j] = W[j, jminus]
        end
        
        # Stop criterion
        if norm(W - Wold,2) < tol
            break 
        end
        Wold = copy(W)
    end
    
    if with_precision == true
        # Build precision matrix
        # We don't use their formula as the entries of W11 got updated over the iterations 
        θ = inv(W)
        return W, θ
    else
        return W
    end
end