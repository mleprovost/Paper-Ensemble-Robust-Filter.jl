export seqassim_loclikenrf

# Write the seqassim routine for the LikEnRF with further options to output the dof estimate over time and store previous joint samples


"""
		seqassim_loclikenrf(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::SeqFilter, X, Ny, Nx, t0::Float64)

Generic API for sequential data assimilation for any sequential filter of parent type `SeqFilter`.
"""
function seqassim_loclikenrf(F::StateSpace, data::SyntheticData, J::Int64, ϵx::InflationType, algo::LocLikEnRF, X, Ny, Nx, t0::Float64)

Ne = size(X, 2)

step = ceil(Int, algo.Δtobs/algo.Δtdyn)

statehist = Array{Float64,2}[]
push!(statehist, deepcopy(X[Ny+1:Ny+Nx,:]))

n0 = ceil(Int64, t0/algo.Δtobs) + 1
Acycle = n0:n0+J-1
tspan = (t0, t0 + algo.Δtobs)
# Define the dynamical system
prob = ODEProblem(F.f, zeros(Nx), tspan)

if algo.estimate_dof == false
	refresh_dof = ceil(Int, algo.Δtrefresh/algo.Δtobs)
	# @show refresh_dof

	Ne_buffer = 600#refresh_dof*Ne

	# @show Ne_buffer
	# @assert Ne_buffer//Ne > refresh_dof "The refreshing is too frequent compared to the ensemble size" 

	buffer_joint = zeros(Ny+Nx, Ne_buffer)
end


νXhist = Float64[]

# Run filtering algorithm
@showprogress for i=1:length(Acycle)
    # Forecast
	tspan = (t0+(i-1)*algo.Δtobs, t0+i*algo.Δtobs)
	prob = remake(prob; tspan=tspan)

	prob_func(prob,i,repeat) = ODEProblem(prob.f, X[Ny+1:Ny+Nx,i],prob.tspan)

	ensemble_prob = EnsembleProblem(prob,output_func = (sol,i) -> (sol[end], false),
	prob_func=prob_func)
	sim = solve(ensemble_prob, Tsit5(), adaptive = true, EnsembleThreads(),trajectories = Ne,
				dense = false, save_everystep=false);

	@inbounds for i=1:Ne
	    X[Ny+1:Ny+Nx, i] .= deepcopy(sim[i])
	end


    # Assimilation # Get real measurement # Fix this later # Things are shifted in data.yt
    ystar = data.yt[:,Acycle[i]]
	# Replace at some point by realobserve(model.h, t0+i*model.Δtobs, ens)
	# Perform inflation for each ensemble member
	ϵx(X, Ny+1, Ny+Nx)

	# Compute measurements
	observe(F.h, X, t0+i*algo.Δtobs, Ny, Nx)

	# @show algo.νX[1]
	if algo.estimate_dof == false 
		# buffer_joint = hcat(buffer_joint, copy(X))

		# Estimate dof

			# Sample from the observational noise
			E = zeros(Ny, Ne)
			if typeof(algo.ϵy) <: AdditiveInflation
				E .= algo.ϵy.σ*randn(Ny, Ne) .+ algo.ϵy.m
			elseif typeof(algo.ϵy) <: TDistAdditiveInflation
				E .= rand(algo.ϵy.dist, Ne)
			else
				print("Inflation type not defined")
			end

			buffer_joint[:,1:Ne] .= deepcopy(X)
			# Add the observation noise to the synthetic observations
			buffer_joint[1:Ny,1:Ne] .+= E

			# We want to recompute the dof if we are at a multiple of refresh_dof 
			# and if we have accumulated Ne_buffer samples in buffer_joint
			if norm(mod(i, refresh_dof))<1e-12 && i*Ne >= Ne_buffer

				νXestimated = EMq(buffer_joint, 0.97, 100; rtol = 1e-5, 
								estimate_dof = true, νX = νX_DEFAULT)[end]
				# print("New estimated νX value: $(νXestimated)")
				# Update dof of algo
				algo = LikEnRF(algo.ϵy, algo.Δtdyn, algo.Δtobs, algo.Δtrefresh;
							estimate_dof = false, νX = [νXestimated])
			end

			# Slides samples in joint to leave room for the next iteration
			buffer_joint[:,Ne+1:Ne_buffer] .= buffer_joint[:,1:Ne_buffer-Ne]
			buffer_joint[:,1:Ne] .= 0.0

		# end
	end


    # Generate posterior samples.
	# Note that the additive inflation of the observation is applied within the sequential filter.

    X = algo(X, ystar, t0+i*algo.Δtobs-t0)

	push!(νXhist, copy(algo.νX[1]))

	# Filter state
	if algo.isfiltered == true
		for i=1:Ne
			statei = view(X, Ny+1:Ny+Nx, i)
			statei .= algo.G(statei)
		end
	end

    push!(statehist, copy(X[Ny+1:Ny+Nx,:]))
	end

return statehist, νXhist
end