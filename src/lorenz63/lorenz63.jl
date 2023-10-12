export setup_lorenz63, benchmark_enkf_lorenz63, benchmark_enrf_lorenz63

function setup_lorenz63(path::String, Ne_array::Array{Int64,1})
    Nx = 3
    Ny = 3
    Δtdyn = 0.05
    Δtobs = 0.1

    σx = 1e-6
    σy = 2.0

    ϵx = AdditiveInflation(Nx, zeros(Nx), σx)
    ϵy = AdditiveInflation(Ny, zeros(Ny), σy)
    tspinup = 200.0
    Tspinup = 2000
    tmetric = 400.0
    Tmetric = 4000
    t0 = 0.0
    tf = 600.0
    Tf = 6000

    Tburn = 2000
    Tstep = Tf - Tburn

    f = lorenz63!
    h(x, t) = x

	F = StateSpace(lorenz63!, h)

    model = Model(Nx, Ny, Δtdyn, Δtobs, ϵx, ϵy, MvNormal(zeros(Nx), Matrix(1.0*I, Nx, Nx)), Tburn, Tstep, Tspinup, F);

    # Set initial condition
    x0 = rand(model.π0)
    # x0 = [0.645811242103507;  -1.465126216973632;   0.385227725149822];

    # Run dynamics and generate data
    data = generate_data(model, x0, model.Tspinup+model.Tstep);


    for Ne in Ne_array
        spin_model(model, data, Ne, path)
    end

    return model, data
end

function benchmark_enkf_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1})
    # @assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"
    
    #Store all the metric per number of ensemble members
    metric_list = []
    
    @showprogress for Ne in Ne_array
        metric_Ne = Metrics[]
        for β in β_array
            @show Ne, β
            # Load file
            X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "state")[end]
    
            X = zeros(model.Ny + model.Nx, Ne)
            X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)
            J = model.Tstep
            t0 = model.Tspinup*model.Δtobs
            F = model.F
    
            # Define the observation matrix to the identity
            H = Matrix(1.0*I, 3, 3)
            idx = [collect(1:3)'; collect(1:3)']
    
            enkf = SeqLiksEnKF(model.ϵy, β, H, model.Δtdyn, model.Δtobs, idx)
    
            # Use additive inflation
            ϵx = AdditiveInflation(model.Nx, zeros(model.Nx), model.ϵx.σ)
    
            @time statehist = seqassim(F, data, J, model.ϵx, enkf, X, model.Ny, model.Nx, t0);
    
            metric = output_metrics(data, model, J, statehist)
            push!(metric_Ne, deepcopy(metric))
            println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(metric.rmse_mean))
        end
        push!(metric_list, deepcopy(metric_Ne))
    end
    
    return metric_list
end


function benchmark_enrf_lorenz63(model::Model, data::SyntheticData, path::String, Ne_array::Array{Int64,1}, β_array::Array{Float64,1})
    # @assert path[1]=='/' && path[end]=='/' "This function expects a / at the extremities of path"
    
    #Store all the metric per number of ensemble members
    metric_list = []
    
    @showprogress for Ne in Ne_array
        metric_Ne = Metrics[]
        for β in β_array
            @show Ne, β
            # Load file
            X0 = load(path*"set_up_Ne"*string(Ne)*".jld", "state")[end]
    
            X = zeros(model.Ny + model.Nx, Ne)
            X[model.Ny+1:model.Ny+model.Nx,:] .= copy(X0)
            J = model.Tstep
            t0 = model.Tspinup*model.Δtobs
            F = model.F
    
            # Define the observation matrix to the identity
            H = Matrix(1.0*I, 3, 3)
            idx = [collect(1:3)'; collect(1:3)']
    
            # The multiplicative inflation is not implemented in the EnRF
            enrf = SeqLikEnRF(model.ϵy, β, H, model.Δtdyn, model.Δtobs, idx)
    
            # Use additive inflation
            ϵx = AdditiveInflation(model.Nx, zeros(model.Nx), model.ϵx.σ)
    
            @time statehist = seqassim(F, data, J, model.ϵx, enrf, X, model.Ny, model.Nx, t0);
    
            metric = output_metrics(data, model, J, statehist)
            push!(metric_Ne, deepcopy(metric))
            println("Ne "*string(Ne)*"& β "*string(β)*" RMSE: "*string(metric.rmse_mean))
        end
        push!(metric_list, deepcopy(metric_Ne))
    end
    
    return metric_list
end
