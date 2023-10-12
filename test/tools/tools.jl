


@testset "Test α function" begin
    for n ∈ [1; 5; 10]
        ν = 2 + rand(1:10)
        x = randn(n)
        μ = randn(n)
        C = randn(n,n)
        C = C*C'
        C += 0.5*I

        αtrue = (ν + (x-μ)'*inv(C)*(x-μ))/(ν + n)
        @test isapprox(αtdist(x, μ, C, ν, n), αtrue, atol = 1e-10)
    end
end
