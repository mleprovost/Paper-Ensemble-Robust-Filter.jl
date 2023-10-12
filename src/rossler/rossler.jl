
export rossler!, rossler2!, generate_rossler

"""
    rossler!(du,u,p,t)

Compute in-place the right-hand-side of the Rossler system for a state `u` at time `t`,
and store it in `du`. `p` is vector of user-defined parameters.
"""
function rossler!(du,u,p,t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + 0.1*u[2]
    du[3] = 0.1 - 9.0*u[3] + u[1]*u[3]
    return du
end

# Supervised machine learning to estimate
# instabilities in chaotic systems: estimation of local
# Lyapunov exponents
function rossler2!(du,u,p,t)
    du[1] = -u[2] - u[3]
    du[2] = u[1] + 0.37*u[2]
    du[3] = 0.2 - 5.7*u[3] + u[1]*u[3]
    return du
end


function generate_rossler(model::Model, x0::Array{Float64,1}, J::Int64)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)
    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = ODEProblem(model.F.f,x,tspan)

    for i=1:J
    	# Run dynamics and save results
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, tspan = tspan)

    	sol = solve(prob, Tsit5(), adaptive = true, dense = false, save_everystep = false)
    	x .= deepcopy(sol.u[end])
    	# for j=1:step
    	# 	t = t0 + (i-1)*algo.Δtobs+(j-1)*algo.Δtdyn
        # 	_, x = model.f(t+(i-1)*model.Δtdyn, x)
    	# end
    	model.ϵx(x)

    	# Collect observations
    	tt[i] = deepcopy(i*model.Δtobs)
    	xt[:,i] = deepcopy(x)
    	yt[:,i] = deepcopy(model.F.h(x, tt[i]))
        # model.ϵy(yt[:,i])
		yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end
