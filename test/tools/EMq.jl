

@testset "Test derivative_rhsEMqν" begin
    νold = 2.1
    Ne = 20
    Nx = 5
    ν = 3.0
    q = 0.85

    wq1 = 0.1 .+ rand(Ne)
    wq2 = 0.1 .+ rand(Ne)
    Wq = 0.1 .+ rand(Ne)

    @test isapprox(ForwardDiff.derivative(x->rhsEMqν(x, q, wq1, wq2, Wq), ν),
                   derivative_rhsEMqν(ν, q, wq1, wq2, Wq),
                   atol = 1e-6)
end

@testset "Test that EMq reverts to EM q = 1.0" begin
    Nx = 10
    ν = 5.0
    μX = zeros(Nx)
    CX = PDiagMat(ones(Nx))
    πX = Distributions.GenericMvTDist(ν, μX, CX)

    Ne = 100

    # We consider a linear Gaussian observation model
    X = zeros(Nx, Ne)
    X .= rand(πX, Ne)
    μXEM, AXEM, νEM = EM(X, 100; rtol = 1e-4)

    μXEMq, AXEMq, νEMq = EMq(X, 1.0, 100; rtol = 1e-4)

    @test isapprox(μXEM, μXEMq, atol = 1e-1)
    @test isapprox(AXEM, AXEMq, atol = 1e-1)
    @test isapprox(νEM, νEMq, atol = 1e-1)

end
