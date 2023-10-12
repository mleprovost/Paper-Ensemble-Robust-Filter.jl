
export αtdist, modfloat

function αtdist(x, μ, C, ν, n::Int64)
    #out = ν + dot(x-μ, Symmetric(C)\(x-μ))
    out = ν + sum(abs2, PDMats.chol_lower(cholesky(Symmetric(C))) \ (x -μ))
    return out/(ν + n)
end

# A function that returns true if a float number a is an integer multiple of a float b

modfloat(a,b) = abs(mod(a + 0.5 * b, b) - 0.5 * b) < 1e-12
