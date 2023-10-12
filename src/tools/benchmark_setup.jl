export generate_data, spin_model, generate_data_SDE

function generate_data(model::Model, x0, J::Int64)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)
    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = ODEProblem(model.F.f,deepcopy(x),tspan)

    @showprogress for i=1:J
    	# Run dynamics and save results
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, u0 = deepcopy(x), tspan = tspan)
    	sol = solve(prob, AutoTsit5(Rosenbrock23()), adaptive = true,
		            dense = false, save_everystep = false)
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

		if typeof(model.ϵy) <: AdditiveInflation
			yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
		elseif typeof(model.ϵy) <: TDistAdditiveInflation
			yt[:,i] .+= rand(model.ϵy.dist)
		else
			print("Inflation type not defined")
		end

		# yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end

function spin_model(model::Model, data::SyntheticData, Ne::Int64, path::String)

	# Set initial condition
	X = zeros(model.Ny + model.Nx, Ne)
	X[model.Ny+1:model.Ny+model.Nx,:] .= rand(model.π0, Ne)#sqrt(model.C0)*randn(model.Nx, Ne) .+ model.m0

	J = model.Tspinup
	t0 = 0.0
	F = model.F

    idx = [collect(1:3)'; collect(1:3)']

    enkf = StochEnKF(model.ϵy, model.Δtdyn, model.Δtobs)

	statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0)

    save(path*"set_up_Ne"*string(Ne)*".jld", "state", statehist, "Ne", Ne, "x0", data.x0, "tt", data.tt, "xt", data.xt, "yt", data.yt)

    # return statehist
	_,_,rmse_mean,_ = metric_hist(rmse, data.xt[:,1:J], statehist[2:end])
	println("Ne "*string(Ne)* " RMSE: "*string(rmse_mean))
	# Save data
	# save(path*"set_up_Ne"*string(Ne)*".jld", "X", statehist[end], "Ne", Ne, "x0", data.x0, "tt", data.tt, "xt", data.xt, "yt", data.yt)
end


function generate_data_SDE(model::Model, x0, J::Int64)

    @assert model.Nx == size(x0,1) "Error dimension of the input"
    xt = zeros(model.Nx,J)
    x = deepcopy(x0)
    yt = zeros(model.Ny,J)
    tt = zeros(J)

    t0 = 0.0

    step = ceil(Int, model.Δtobs/model.Δtdyn)
    tspan = (t0, t0 + model.Δtobs)
    prob = SDEProblem(model.F.f, model.F.g, deepcopy(x), tspan)

	# de = modelingtoolkitize(prob)
	# prob = ODEProblem(de,x,tspan)


# Note jac=true,sparse=true makes it automatically build sparse Jacobian code
# as well!
# fastprob = ODEProblem(de, [], (0.0, 0.1), jac = true, sparse = true)

    @showprogress for i=1:J
    	# Run dynamics and save results
    	tspan = (t0 + (i-1)*model.Δtobs, t0 + i*model.Δtobs)
    	prob = remake(prob, u0 = deepcopy(x), tspan = tspan)
    	sol = solve(prob,  StochasticDiffEq.SKenCarp(), dt = model.Δtobs,# adaptive = true,
		            dense = false, save_everystep = false)
    	x .= deepcopy(sol.u[end])
    	# for j=1:step
    	# 	t = t0 + (i-1)*algo.Δtobs+(j-1)*algo.Δtdyn
        # 	_, x = model.f(t+(i-1)*model.Δtdyn, x)
    	# end
    	# model.ϵx(x)

    	# Collect observations
    	tt[i] = deepcopy(i*model.Δtobs)
    	xt[:,i] = deepcopy(x)
		yt[:,i] = deepcopy(model.F.h(x, tt[i]))

		if typeof(model.ϵy) <: AdditiveInflation
			yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
		elseif typeof(model.ϵy) <: TDistAdditiveInflation
			yt[:,i] .+= rand(model.ϵy.dist)
		else
			print("Inflation type not defined")
		end

		# yt[:,i] .+= model.ϵy.m + model.ϵy.σ*randn(model.Ny)
    end
    	return SyntheticData(tt, model.Δtdyn, x0, xt, yt)
end
