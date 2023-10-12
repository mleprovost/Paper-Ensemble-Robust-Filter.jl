

@testset "Test glasso algorithm" begin
    
    
    # Distributions.jl takes as input the scale matrix C
    Nx = 10
    μX = zeros(Nx)
    ΣX = PDiagMat(ones(Nx))
    πX = Distributions.MvNormal(μX, ΣX)
    
    Ne = 50
    X = rand(πX, Ne)
    S = cov(X')
    
    W, θ = glasso(S, 0.01; tol = 1e-3, with_precision = true)

    @test isapprox(θ, inv(W), atol = 1e4*eps())

    j = rand(1:Nx)
    jminus = setdiff(1:Nx, j)
    @test isapprox(W[jminus, jminus]*θ[j, jminus] + W[j, jminus]*θ[j, j], zeros(Nx-1), atol= 1e4*eps())
    @test isapprox(W[j, jminus]'*θ[j, jminus] + W[j, j]*θ[j, j], 1.0, atol = 1e4*eps())
end

@testset "Test glasso_block algorithm I, all the components are connected" begin
    
    # Check all the components are connected
    # Distributions.jl takes as input the scale matrix C
    Nx = 5
    μX = zeros(Nx)
    ΣX = PDiagMat(ones(Nx))
    πX = Distributions.MvNormal(μX, ΣX)
    
    Ne = 50
    X = rand(πX, Ne)
    S = cov(X')
    
    Wglasso, θglasso = glasso(S, 0.01; tol = 1e-3, with_precision = true)
    Wblock, θblock = glasso_block(S, 0.01; tol = 1e-3, with_precision = true)


    @test isapprox(θblock, inv(Wblock), atol = 1e4*eps())
    
    # Check that glasso and glasso_block are consistent when all the components are fully connected
    @test isapprox(Wblock, Wglasso, atol = 1e-8)
    @test isapprox(θblock, θglasso, atol = 1e-8)

end

@testset "Test glasso_block algorithm II, not all the components are connected" begin
    
    # Check all the components are connected
    # Distributions.jl takes as input the scale matrix C
    Nx = 40
    μX = zeros(Nx)
    ΣX = PDiagMat(ones(Nx))
    πX = Distributions.MvNormal(μX, ΣX)
    
    Ne = 50
    X = rand(πX, Ne)
    S = cov(X')
    
    Wglasso, θglasso = glasso(S, 0.3; tol = 1e-3, with_precision = true)
    Wblock, θblock = glasso_block(S, 0.3; tol = 1e-3, with_precision = true)


    @test isapprox(θblock, inv(Wblock), atol = 1e4*eps())
    
    # Check that glasso and glasso_block are consistent when not all the components are fully connected
    @test isapprox(Wblock, Wglasso, atol = 1e-8)
    @test isapprox(θblock, θglasso, atol = 1e-8)

end

@testset "Test output of glasso I" begin
    # We verify our implementation of glasso against the routine sklearn.covariance.graphical_lasso
    # implemented in Python 
    # sklearn.covariance.graphical_lasso does not regularize the diagonal entries.
    # For consistent comparisons with our routines, the input S is replaced by S + ρ*I in the routine:
    # sklearn.covariance.graphical_lasso(S + ρ*np.identity(S.shape[0]), ρ, tol = 1e-6)

    S = [6.83423116e-01 -1.47104663e-02  1.65067517e-01 -3.66876552e-04;
            -1.47104663e-02  4.10547381e-01 -3.17069124e-02 -4.80670424e-02;
            1.65067517e-01 -3.17069124e-02  2.83309261e-01  7.89374654e-02;
            -3.66876552e-04 -4.80670424e-02  7.89374654e-02  5.81054718e-01]

    W01_scikit = [0.78342312 0.         0.06506752 0.;
    0.         0.51054738 0.         0.;
    0.06506752 0.         0.38330926 0.;  
    0.         0.         0.         0.68105472]

    θ01_scikit = [ 1.29470323  0.         -0.21977847  0.        ;
               0.          1.95868207  0.          0.        ;
              -0.21977847  0.          2.64616732 -0.        ;
               0.          0.         -0.          1.4683108 ]

    W05_scikit =    [1.18342312 0.         0.         0.        ;
    0.         0.91054738 0.         0.        ;
    0.         0.         0.78330926 0.        ;
    0.         0.         0.         1.08105472] 
    
    
    θ05_scikit = [0.84500631  0.         -0.          0.       ;
              0.          1.09824049  0.          0.        ;
             -0.          0.          1.27663498 -0.        ;
              0.          0.         -0.          0.92502256]

    W01, θ01 = glasso(S, 0.1; tol=1e-6, with_precision=true)
    W05, θ05 = glasso(S, 0.5; tol=1e-6, with_precision=true)

    @test isapprox(W01, W01_scikit, atol = 1e-5)
    @test isapprox(θ01, θ01_scikit, atol = 1e-5)

    @test isapprox(W05, W05_scikit, atol = 1e-5)
    @test isapprox(θ05, θ05_scikit, atol = 1e-5)
end


@testset "Test output of glasso with randomly sparse PD precision matrix II" begin
    # We verify our implementation of glasso against the routine sklearn.covariance.graphical_lasso
    # implemented in Python 
    # sklearn.covariance.graphical_lasso does not regularize the diagonal entries.
    # For consistent comparisons with our routines, the input S is replaced by S + ρ*I in the routine:
    # sklearn.covariance.graphical_lasso(S + ρ*np.identity(S.shape[0]), ρ, tol = 1e-6)

    S_sparse = [2.15580536e-01  2.01744319e-02  3.00019217e-02  2.65082070e-02 4.66648006e-02 -2.19170903e-03 -7.90003693e-02 -5.97199077e-02 -1.07257006e-01 -3.60203650e-02;
    2.01744319e-02  2.40958713e-01  5.74019975e-03 -2.32832417e-03 3.82698998e-02 -5.62470271e-02 -4.72489929e-03 -2.76710237e-02 -1.54848037e-03 -3.01374098e-02;
    3.00019217e-02  5.74019975e-03  2.89755378e-01  9.02767576e-03 1.55652884e-04 -8.23826657e-03 -8.42309580e-03  1.21624792e-02 2.09213740e-03 -1.27891488e-02;
    2.65082070e-02 -2.32832417e-03  9.02767576e-03  3.11016436e-01 -7.48802790e-02  7.80124256e-03 -3.62284759e-02  1.03949864e-02 -6.15889941e-02 -4.98888230e-02;
    4.66648006e-02  3.82698998e-02  1.55652884e-04 -7.48802790e-02 4.26806277e-01 -7.73586157e-02 -1.23740344e-01 -1.02949972e-01 -1.22633725e-01 -6.61689058e-02;
    -2.19170903e-03 -5.62470271e-02 -8.23826657e-03  7.80124256e-03 -7.73586157e-02  1.93483709e-01  2.69691802e-02  1.33614180e-02 -5.28071227e-03  1.99073865e-02;
    -7.90003693e-02 -4.72489929e-03 -8.42309580e-03 -3.62284759e-02 -1.23740344e-01  2.69691802e-02  3.15056662e-01  3.92374058e-02 6.95866577e-02  4.98708787e-02;
    -5.97199077e-02 -2.76710237e-02  1.21624792e-02  1.03949864e-02 -1.02949972e-01  1.33614180e-02  3.92374058e-02  2.47037782e-01 7.86880255e-02  1.08070762e-02;
    -1.07257006e-01 -1.54848037e-03  2.09213740e-03 -6.15889941e-02 -1.22633725e-01 -5.28071227e-03  6.95866577e-02  7.86880255e-02 3.00630416e-01  3.04929438e-02;
    -3.60203650e-02 -3.01374098e-02 -1.27891488e-02 -4.98888230e-02 -6.61689058e-02  1.99073865e-02  4.98708787e-02  1.08070762e-02 3.04929438e-02  2.95176723e-01]

        
    W_sparse_01_scikit = [ 3.15580536e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00 4.09986536e-04  0.00000000e+00 -1.84759024e-05 -2.29581348e-06 -7.25700597e-03  0.00000000e+00;
    0.00000000e+00  3.40958713e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  3.89755378e-01  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  4.11016436e-01 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    4.09986536e-04  0.00000000e+00  0.00000000e+00  0.00000000e+00 5.26806277e-01  0.00000000e+00 -2.37403439e-02 -2.94997237e-03 -2.26337249e-02  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  2.93483709e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    -1.84759024e-05  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.37403439e-02  0.00000000e+00  4.15056662e-01  1.32939492e-04 1.01998104e-03  0.00000000e+00;
    -2.29581348e-06  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.94997237e-03  0.00000000e+00  1.32939492e-04  3.47037782e-01 1.26742725e-04  0.00000000e+00;
    -7.25700597e-03  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.26337249e-02  0.00000000e+00  1.01998104e-03  1.26742725e-04 4.00630416e-01  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  3.95176723e-01]

    θ_sparse_01_scikit =  [3.17008371 -0.         -0.         -0.         -0.          0. 0.          0.          0.05742279  0.;
    -0.          2.93290642 -0.          0.         -0.          0. 0.          0.          0.          0.        ;
    -0.         -0.          2.56571187 -0.         -0.          0. 0.         -0.         -0.          0.        ;
    -0.          0.         -0.          2.43299273  0.         -0. 0.         -0.          0.          0.        ;
    -0.         -0.         -0.          0.          1.9078456   0. 0.1088553   0.01613656  0.10750201  0.        ;
    0.          0.          0.         -0.          0.          3.40734416 -0.         -0.          0.         -0.        ;
    0.          0.          0.          0.          0.1088553  -0. 2.41553589 -0.         -0.         -0.        ;
    0.          0.         -0.         -0.          0.01613656 -0. -0.          2.8816678  -0.         -0.        ;
    0.05742279  0.         -0.          0.          0.10750201  0. -0.         -0.          2.50317961 -0.        ;
    0.          0.          0.          0.          0.         -0. -0.         -0.         -0.          2.53051342]

    @time W_sparse_01, θ_sparse_01 = glasso(S_sparse, 0.1; tol=1e-6, with_precision=true)

    @test isapprox(W_sparse_01, W_sparse_01_scikit, atol = 1e-5)
    @test isapprox(θ_sparse_01, θ_sparse_01_scikit, atol = 1e-5)
end


@testset "Test output of glasso_block I" begin
    # We verify our implementation of glasso against the routine sklearn.covariance.graphical_lasso
    # implemented in Python 
    # sklearn.covariance.graphical_lasso does not regularize the diagonal entries.
    # For consistent comparisons with our routines, the input S is replaced by S + ρ*I in the routine:
    # sklearn.covariance.graphical_lasso(S + ρ*np.identity(S.shape[0]), ρ, tol = 1e-6)

    S = [6.83423116e-01 -1.47104663e-02  1.65067517e-01 -3.66876552e-04;
            -1.47104663e-02  4.10547381e-01 -3.17069124e-02 -4.80670424e-02;
            1.65067517e-01 -3.17069124e-02  2.83309261e-01  7.89374654e-02;
            -3.66876552e-04 -4.80670424e-02  7.89374654e-02  5.81054718e-01]

    W01_scikit = [0.78342312 0.         0.06506752 0.;
    0.         0.51054738 0.         0.;
    0.06506752 0.         0.38330926 0.;  
    0.         0.         0.         0.68105472]

    θ01_scikit = [ 1.29470323  0.         -0.21977847  0.        ;
               0.          1.95868207  0.          0.        ;
              -0.21977847  0.          2.64616732 -0.        ;
               0.          0.         -0.          1.4683108 ]

    W05_scikit =    [1.18342312 0.         0.         0.        ;
    0.         0.91054738 0.         0.        ;
    0.         0.         0.78330926 0.        ;
    0.         0.         0.         1.08105472] 
    
    
    θ05_scikit = [0.84500631  0.         -0.          0.       ;
              0.          1.09824049  0.          0.        ;
             -0.          0.          1.27663498 -0.        ;
              0.          0.         -0.          0.92502256]

    W01, θ01 = glasso_block(S, 0.1; tol=1e-6, with_precision=true)
    W05, θ05 = glasso_block(S, 0.5; tol=1e-6, with_precision=true)

    @test isapprox(W01, W01_scikit, atol = 1e-5)
    @test isapprox(θ01, θ01_scikit, atol = 1e-5)

    @test isapprox(W05, W05_scikit, atol = 1e-5)
    @test isapprox(θ05, θ05_scikit, atol = 1e-5)
end

@testset "Test output of glasso_block with randomly sparse PD precision matrix II" begin
    # We verify our implementation of glasso against the routine sklearn.covariance.graphical_lasso
    # implemented in Python 
    # sklearn.covariance.graphical_lasso does not regularize the diagonal entries.
    # For consistent comparisons with our routines, the input S is replaced by S + ρ*I in the routine:
    # sklearn.covariance.graphical_lasso(S + ρ*np.identity(S.shape[0]), ρ, tol = 1e-6)

    S_sparse = [2.15580536e-01  2.01744319e-02  3.00019217e-02  2.65082070e-02 4.66648006e-02 -2.19170903e-03 -7.90003693e-02 -5.97199077e-02 -1.07257006e-01 -3.60203650e-02;
    2.01744319e-02  2.40958713e-01  5.74019975e-03 -2.32832417e-03 3.82698998e-02 -5.62470271e-02 -4.72489929e-03 -2.76710237e-02 -1.54848037e-03 -3.01374098e-02;
    3.00019217e-02  5.74019975e-03  2.89755378e-01  9.02767576e-03 1.55652884e-04 -8.23826657e-03 -8.42309580e-03  1.21624792e-02 2.09213740e-03 -1.27891488e-02;
    2.65082070e-02 -2.32832417e-03  9.02767576e-03  3.11016436e-01 -7.48802790e-02  7.80124256e-03 -3.62284759e-02  1.03949864e-02 -6.15889941e-02 -4.98888230e-02;
    4.66648006e-02  3.82698998e-02  1.55652884e-04 -7.48802790e-02 4.26806277e-01 -7.73586157e-02 -1.23740344e-01 -1.02949972e-01 -1.22633725e-01 -6.61689058e-02;
    -2.19170903e-03 -5.62470271e-02 -8.23826657e-03  7.80124256e-03 -7.73586157e-02  1.93483709e-01  2.69691802e-02  1.33614180e-02 -5.28071227e-03  1.99073865e-02;
    -7.90003693e-02 -4.72489929e-03 -8.42309580e-03 -3.62284759e-02 -1.23740344e-01  2.69691802e-02  3.15056662e-01  3.92374058e-02 6.95866577e-02  4.98708787e-02;
    -5.97199077e-02 -2.76710237e-02  1.21624792e-02  1.03949864e-02 -1.02949972e-01  1.33614180e-02  3.92374058e-02  2.47037782e-01 7.86880255e-02  1.08070762e-02;
    -1.07257006e-01 -1.54848037e-03  2.09213740e-03 -6.15889941e-02 -1.22633725e-01 -5.28071227e-03  6.95866577e-02  7.86880255e-02 3.00630416e-01  3.04929438e-02;
    -3.60203650e-02 -3.01374098e-02 -1.27891488e-02 -4.98888230e-02 -6.61689058e-02  1.99073865e-02  4.98708787e-02  1.08070762e-02 3.04929438e-02  2.95176723e-01]

        
    W_sparse_01_scikit = [ 3.15580536e-01  0.00000000e+00  0.00000000e+00  0.00000000e+00 4.09986536e-04  0.00000000e+00 -1.84759024e-05 -2.29581348e-06 -7.25700597e-03  0.00000000e+00;
    0.00000000e+00  3.40958713e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  3.89755378e-01  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  4.11016436e-01 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    4.09986536e-04  0.00000000e+00  0.00000000e+00  0.00000000e+00 5.26806277e-01  0.00000000e+00 -2.37403439e-02 -2.94997237e-03 -2.26337249e-02  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  2.93483709e-01  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00;
    -1.84759024e-05  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.37403439e-02  0.00000000e+00  4.15056662e-01  1.32939492e-04 1.01998104e-03  0.00000000e+00;
    -2.29581348e-06  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.94997237e-03  0.00000000e+00  1.32939492e-04  3.47037782e-01 1.26742725e-04  0.00000000e+00;
    -7.25700597e-03  0.00000000e+00  0.00000000e+00  0.00000000e+00 -2.26337249e-02  0.00000000e+00  1.01998104e-03  1.26742725e-04 4.00630416e-01  0.00000000e+00;
    0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00 0.00000000e+00  3.95176723e-01]

    θ_sparse_01_scikit =  [3.17008371 -0.         -0.         -0.         -0.          0. 0.          0.          0.05742279  0.;
    -0.          2.93290642 -0.          0.         -0.          0. 0.          0.          0.          0.        ;
    -0.         -0.          2.56571187 -0.         -0.          0. 0.         -0.         -0.          0.        ;
    -0.          0.         -0.          2.43299273  0.         -0. 0.         -0.          0.          0.        ;
    -0.         -0.         -0.          0.          1.9078456   0. 0.1088553   0.01613656  0.10750201  0.        ;
    0.          0.          0.         -0.          0.          3.40734416 -0.         -0.          0.         -0.        ;
    0.          0.          0.          0.          0.1088553  -0. 2.41553589 -0.         -0.         -0.        ;
    0.          0.         -0.         -0.          0.01613656 -0. -0.          2.8816678  -0.         -0.        ;
    0.05742279  0.         -0.          0.          0.10750201  0. -0.         -0.          2.50317961 -0.        ;
    0.          0.          0.          0.          0.         -0. -0.         -0.         -0.          2.53051342]

    @time W_sparse_01, θ_sparse_01 = glasso_block(S_sparse, 0.1; tol=1e-6, with_precision=true)

    @test isapprox(W_sparse_01, W_sparse_01_scikit, atol = 1e-5)
    @test isapprox(θ_sparse_01, θ_sparse_01_scikit, atol = 1e-5)
end