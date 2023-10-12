export advection!


"""
    advection!(du,u,p,t)

Compute in-place the right-hand-side of the advection-diffusion problem in 1D for a state `u` at time `t`,
and store it in `du`. `p` is vector of user-defined parameters.
"""
function advection!(du,u,p,t)
    c= p["Velocity"]
    ν = p["Nu"]
    μ = p["Mu"]
    Δx = p["Deltax"]

    n = size(u,1)
    # Periodic boundaries u[0] = u[n] and u[n+1] = u[1]
    du[1] = -c/(2*Δx)*(u[2] - u[n]) - ν*u[1] + (μ/Δx^2)*(u[2] - 2*u[1] + u[n])
    du[n] = -c/(2*Δx)*(u[1] - u[n-1]) - ν*u[n] + (μ/Δx^2)*(u[1] - 2*u[n] + u[n-1])

    @inbounds for i=2:n-1
        du[i] = -c/(2*Δx)*(u[i+1] - u[i-1]) - ν*u[i] + (μ/Δx^2)*(u[i+1] - 2*u[i] + u[i-1])
    end
    return du
end
