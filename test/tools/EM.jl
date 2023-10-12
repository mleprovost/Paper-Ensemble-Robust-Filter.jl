

@testset "Test derivative_ϕtdist" begin
    x = 1.0 + rand()

    @test isapprox(ForwardDiff.derivative(RobustFilter.ϕtdist, x),
                   RobustFilter.derivative_ϕtdist(x), atol = 1e-6)

end


@testset "Test derivative_rhsEMν" begin
    νold = 2.1
    Ne = 20
    γold = 0.1 .+ rand(Ne)
    Nx = 5
    ν = 3.0

    @test isapprox(ForwardDiff.derivative(x->rhsEMν(x, νold, γold, Nx), ν),
                   derivative_rhsEMν(ν, νold, γold, Nx),
                   atol = 1e-6)
end

@testset "Test derivative_rhsaEMν" begin
    νold = 2.1
    Ne = 20
    δ = 0.1 .+ rand(Ne)
    Nx = 5
    ν = 3.0

    @test isapprox(ForwardDiff.derivative(x->rhsaEMν(x, νold, δ, Nx), ν),
                   derivative_rhsaEMν(ν, νold, δ, Nx),
                   atol = 1e-6)
end

@testset "Test derivative_rhsECMEν" begin
    νold = 2.1
    Ne = 20
    δ = 0.1 .+ rand(Ne)
    Nx = 5
    ν = 3.0

    @test isapprox(ForwardDiff.derivative(x->rhsECMEν(x, δ, Nx), ν),
                   derivative_rhsECMEν(ν, δ, Nx),
                   atol = 1e-6)
end
