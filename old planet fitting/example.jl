using Pkg
Pkg.activate(".")
# Pkg.add("GPLinearODEMaker")
# Pkg.add("DataFrames")
# Pkg.add("CSV")
# Pkg.add("Statistics")
# Pkg.add("LinearAlgebra")
# Pkg.add("Distributed")
# Pkg.add("Unitful")
# Pkg.add("UnitfulAstro")
# Pkg.add("PyPlot")
# Pkg.add("PyCall")
# Pkg.add("SpecialFunctions")
# Pkg.add("Dates")
# Pkg.add("Optim")
# Pkg.add("LineSearches")
# Pkg.add("Random")
Pkg.instantiate()

include("src/all_functions.jl")

using DataFrames
using CSV
using Statistics
import GPLinearODEMaker
GLOM = GPLinearODEMaker

called_from_terminal = length(ARGS) > 0

###################################
# Loading data and setting kernel #
###################################

kernel_names = ["pp", "se", "m52", "qp", "m52_m52", "se_se"]
star_rot_rate = 130.  # days
initial_hypers = [[star_rot_rate], [star_rot_rate], [star_rot_rate], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1]]

# if called from terminal with an argument, use a full dataset. Otherwise, use a smaller testing set
called_from_terminal ? kernel_choice = parse(Int, ARGS[1]) : kernel_choice = 3
kernel_name = kernel_names[kernel_choice]

kernel_function, num_kernel_hyperparameters = GLOM.include_kernel(kernel_name)

data = NaN# TODO

xs = collect(data.BJD)
xs .-= mean(xs)
ys = collect(Iterators.flatten(zip(data.RV, data.NaD1 + data.NaD2, data.Halpha)))
noise = collect(Iterators.flatten(zip(data.e_RV, sqrt.(data.e_NaD1.^2 + data.e_NaD2.^2), data.e_Halpha)))

results_dir = "results/"
isdir(results_dir) || mkpath(results_dir)

problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, 3, 3, xs, ys; noise = noise, a0=[[1. 1 0];[1 0 1];[1 0 1]])
# If all a's active:
# problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, 3, 3, xs, ys; noise = noise, a0=[[1. 1 1];[1 1 1];[1 1 1]])
GLOM.normalize_problem_definition!(problem_definition)
problem_definition_rv = GLO_RV(problem_definition)

if kernel_name in ["pp", "se", "m52"]
    # normals
    parameters = gamma_mode_std_2_alpha_theta(star_rot_rate, star_rot_rate / 2)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], parameters; d=d)]
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 1
        hps .*= centered_rand(rng, length(hps); center=1, scale=0.5)
        return hps
    end
elseif  kernel_name == "qp"
    # qp
    paramsλp = gamma_mode_std_2_alpha_theta(1, 0.4)
    σP = star_rot_rate / 2; σse = star_rot_rate / 2; ρ = .9
    Σ_qp_prior = bvnormal_covariance(σP, σse, ρ)
    μ_qp_prior = [star_rot_rate, star_rot_rate * 2]
    function nlogprior_kernel_hyperparameters(n_kern_hyper::Integer, total_hyperparameters::Vector{<:Real}, d::Integer)
        hps = total_hyperparameters[(end - (n_kern_hyper - 1)):end]
        if d == 0
            return -(log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.]) + log_gamma(hps[3], paramsλp))
        elseif d == 1
            return append!(zeros(length(total_hyperparameters) - n_kern_hyper), -[log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,0]), log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,1]), log_gamma(hps[3], paramsλp; d=d)])
        elseif d == 2
            H = zeros(length(total_hyperparameters), length(total_hyperparameters))
            H[end-2,end-2] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[2,0])
            H[end-1,end-1] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[0,2])
            H[end,end] = log_gamma(hps[3], paramsλp; d=d)
            H[end-2,end-1] = log_bvnormal(hps[1:2], Σ_qp_prior; μ=μ_qp_prior, lows=[0,0.], d=[1,1])
            return Symmetric(-H)
        end
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 3
        hps[1] *= centered_rand(rng; center=1.0, scale=0.4)
        hps[2] *= centered_rand(rng; center=1.2, scale=0.4)
        hps[3] *= centered_rand(rng; center=0.8, scale=0.4)
        return hps
    end
elseif kernel_name in ["se_se", "m52_m52"]
    # m52_m52
    paramsλ1 = gamma_mode_std_2_alpha_theta(star_rot_rate, star_rot_rate / 2)
    paramsλ2 = gamma_mode_std_2_alpha_theta(2 * star_rot_rate, star_rot_rate / 2)
    function kernel_hyper_priors(hps::Vector{<:Real}, d::Integer)
        return [log_gamma(hps[1], paramsλ1; d=d), log_gamma(hps[2], paramsλ2; d=d), log_gaussian(hps[3], [1, 1]; d=d)]
    end
    function add_kick!(hps::Vector{<:Real})
        @assert length(hps) == 3
        if hps[1] > hps[2]; hps[3] = 1 / hps[3] end
        hold = sort(hps[1:2])
        hps[1] = hold[1] * centered_rand(rng; center=0.8, scale=0.4)
        hps[2] = hold[2] * centered_rand(rng; center=1.2, scale=0.4)
        hps[3] *= centered_rand(rng; center=1, scale=0.4)
        return hps
    end
end

if !(kernel_name in ["qp", "white"])
    function nlogprior_kernel_hyperparameters(n_kern_hyper::Integer, total_hyperparameters::Vector{<:Real}, d::Integer)
        hps = total_hyperparameters[(end - (n_kern_hyper - 1)):end]
        if d == 0
            return -sum(kernel_hyper_priors(hps, d))
        elseif d == 1
            return append!(zeros(length(total_hyperparameters) - n_kern_hyper), -kernel_hyper_priors(hps, d))
        elseif d == 2
            H = zeros(length(total_hyperparameters), length(total_hyperparameters))
            H[(end - (n_kern_hyper - 1)):end, (end - (n_kern_hyper - 1)):end] -= Diagonal(kernel_hyper_priors(hps, d))
            return H
        end
    end
end

total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
append!(total_hyperparameters, initial_hypers[kernel_choice])

possible_labels = [
    [L"\lambda_{pp}"],
    [L"\lambda_{se}"],
    [L"\lambda_{m52}"],
    [L"\tau_p" L"\lambda_{se}" L"^1/_{\lambda_{p}}"],
    [L"\lambda_{m52_1}" L"\lambda_{m52_2}" L"\sqrt{ratio}"],
    [L"\lambda_{se_1}" L"\lambda_{se_2}" L"\sqrt{ratio}"]]

workspace = GLOM.nlogL_matrix_workspace(problem_definition, total_hyperparameters)

using Optim

# storing initial hyperparameters
initial_x = GLOM.remove_zeros(total_hyperparameters)

f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)
g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)
h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)

function f_no_print(non_zero_hyper::Vector{T}) where {T<:Real}
    nprior = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0)
    if nprior == Inf
        return nprior
    else
        return f_no_print_helper(non_zero_hyper) + nprior
    end
end

function f(non_zero_hyper::Vector{T}) where {T<:Real}
    println(non_zero_hyper)
    global current_hyper[:] = non_zero_hyper
    return f_no_print(non_zero_hyper)
end

function g!(G::Vector{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    if nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 0) == Inf
        G[:] .= 0
    else
        global current_hyper[:] = non_zero_hyper
        G[:] = g!_helper(non_zero_hyper) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 1)
    end
end

function h!(H::Matrix{T}, non_zero_hyper::Vector{T}) where {T<:Real}
    H[:, :] = h!_helper(non_zero_hyper) + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, non_zero_hyper, 2)
end

# ends optimization if true
function optim_cb(x::OptimizationState)
    println()
    if x.iteration > 0
        println("Iteration:              ", x.iteration)
        println("Time so far:            ", x.metadata["time"], " s")
        println("Unnormalized posterior: ", x.value)
        println("Gradient norm:          ", x.g_norm)
        println()
    end
    return false
end


function do_gp_fit_gridsearch!(non_zero_hyper::Vector{<:Real}, ind::Integer)
    println("gridsearching over hp$ind")
    current_hp = non_zero_hyper[ind]
    spread = max(0.5, abs(current_hp) / 4)
    possible_hp = range(current_hp - spread, current_hp + spread, length=11)
    new_hp = current_hp
    new_f = f_no_print(non_zero_hyper)
    println("starting at hp = ", new_hp, " -> ", new_f)
    println("searching from ", current_hp - spread, " to ", current_hp + spread)
    for hp in possible_hp
        hold = copy(non_zero_hyper)
        hold[ind] = hp
        possible_f = f_no_print(hold)
        if possible_f < new_f
            new_hp = hp
            new_f = possible_f
        end
    end
    non_zero_hyper[ind] = new_hp
    println("ending at hp   = ", new_hp, " -> ", new_f)
    return non_zero_hyper
end


function fit_GP!(initial_x::Vector{<:Real}; g_tol=1e-6, iterations=200)
    time0 = Libc.time()
    attempts = 0
    in_saddle = true
    global current_hyper = copy(initial_x)
    while attempts < 10 && in_saddle
        attempts += 1
        if attempts > 1;
            println("found saddle point. starting attempt $attempts with a perturbation")
            global current_hyper[1:end-problem_definition.n_kern_hyper] += centered_rand(rng, length(current_hyper) - problem_definition.n_kern_hyper)
            global current_hyper[end-problem_definition.n_kern_hyper+1:end] = add_kick!(current_hyper[end-problem_definition.n_kern_hyper+1:end])
        end
        if kernel_name == "qp"
            gridsearch_every = 100
            iterations = Int(ceil(iterations * 1.5))
            converged = false
            i = 0
            before_grid = zeros(length(current_hyper))
            global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
            try
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            catch
                println("retrying fit")
                i = 0
                while i < Int(ceil(iterations / gridsearch_every)) && !converged
                    global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=gridsearch_every)) # 27s
                    before_grid[:] = current_hyper
                    global current_hyper = do_gp_fit_gridsearch!(current_hyper, length(current_hyper) - 2)
                    converged = result.g_converged && isapprox(before_grid, current_hyper)
                    i += 1
                end
            end
        else
            try
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations)) # 27s
            catch
                println("retrying fit")
                global result = optimize(f, g!, h!, current_hyper, NewtonTrustRegion(), Optim.Options(callback=optim_cb, g_tol=g_tol, iterations=iterations))
            end
        end
        current_det = det(h!(zeros(length(initial_x), length(initial_x)), current_hyper))
        println(current_det)
        in_saddle = current_det <= 0
    end
    return Libc.time() - time0
end

time1 = fit_GP!(initial_x)
println(result)

fit1_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(problem_definition, result.minimizer)

############
# Post fit #
############

println("starting hyperparameters")
println(total_hyperparameters)
initial_nlogL = GLOM.nlogL_GLOM(problem_definition, total_hyperparameters)
initial_uE = -initial_nlogL - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nlogL1 = GLOM.nlogL_GLOM!(workspace, problem_definition, fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters, 0)
println(uE1, "\n")

#########################
# Keplerian periodogram #
#########################

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
freq_grid = autofrequency(problem_definition_rv.time; samples_per_peak=11)
period_grid = 1 ./ reverse(freq_grid)
amount_of_periods = length(period_grid)

Σ_obs = GLOM.Σ_observations(problem_definition, fit1_total_hyperparameters)

# making necessary variables local to all workers
fit1_total_hyperparameters_nz = GLOM.remove_zeros(fit1_total_hyperparameters)
nlogprior_kernel = nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters_nz, 0)

auto_addprocs()
@everywhere import Pkg
@everywhere Pkg.activate(".")
@everywhere include("src/all_functions.jl")

GLOM.sendto(workers(), kernel_name=kernel_name)
@everywhere GLOM.include_kernel(kernel_name)

GLOM.sendto(workers(), problem_definition_rv=problem_definition_rv, fit1_total_hyperparameters=fit1_total_hyperparameters, Σ_obs=Σ_obs, nlogprior_kernel=nlogprior_kernel)
@everywhere function fit_kep_hold_P(P::Unitful.Time)  # 40x slower than just epicyclic fit. neither optimized including priors
    ks = fit_kepler(problem_definition_rv, Σ_obs, kep_signal_epicyclic(P=P))
    return fit_kepler(problem_definition_rv, Σ_obs, kep_signal_wright(maximum([0.1u"m/s", ks.K]), ks.P, ks.M0, minimum([ks.e, 0.3]), ks.ω, ks.γ); print_stuff=false, hold_P=true, avoid_saddle=false)
end
@everywhere function kep_unnormalized_evidence_distributed(P::Unitful.Time)  # 40x slower than just epicyclic fit. neither optimized including priors
    ks = fit_kep_hold_P(P)
    if ks==nothing
        return [-Inf, -Inf]
    else
        val = GLOM.nlogL_GLOM(
            problem_definition_rv.GLO,
            fit1_total_hyperparameters;
            Σ_obs=Σ_obs,
            y_obs=remove_kepler(problem_definition_rv, ks))
        return [-val, logprior_kepler(ks; use_hk=true) - nlogprior_kernel - val]
    end
end

# parallelize with DistributedArrays
pmap(x->kep_unnormalized_evidence_distributed(x), collect(1.:nworkers()) * u"d", batch_size=1)
@elapsed holder = pmap(x->kep_unnormalized_evidence_distributed(x), period_grid, batch_size=Int(floor(amount_of_periods / nworkers()) + 1))
likelihoods = [holder[i][1] for i in 1:length(holder)]
unnorm_evidences = [holder[i][2] for i in 1:length(holder)]

# @time holder_lin = collect(map(kep_unnormalized_evidence_lin_distributed, period_grid_dist))
# likelihoods_lin = [holder_lin[i][1] for i in 1:length(holder_lin)]
# unnorm_evidences_lin = [holder_lin[i][2] for i in 1:length(holder_lin)]
best_periods = period_grid[find_modes(unnorm_evidences; amount=10)]
best_period = best_periods[1]

println("found period:    $(ustrip(best_period)) days")

####################################################################################################
# Refitting GP with full planet signal at found period subtracted (K,P,ω,γ-linear, M0,e-nonlinear) #
####################################################################################################

f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)

begin
    time0 = Libc.time()
    global current_hyper = GLOM.remove_zeros(fit1_total_hyperparameters)
    global current_ks = fit_kep_hold_P(best_period)
    println("before wright fit: ", kep_parms_str(current_ks))
    # det(∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, current_ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1]))

    results = [Inf, Inf]
    result_change = Inf
    global num_iter = 0
    while result_change > 1e-4 && num_iter < 30
        global current_ks = fit_kepler(problem_definition_rv, workspace.Σ_obs, current_ks; print_stuff=false, avoid_saddle=false)
        println(kep_parms_str(current_ks))
        global current_y = remove_kepler(problem_definition_rv, current_ks)
        fit_GP!(current_hyper)
        results[:] = [results[2], copy(result.minimum)]
        thing = results[1] - results[2]
        if thing < 0; @warn "result increased occured on iteration $num_iter" end
        global result_change = abs(thing)
        global num_iter += 1
        println("change on joint fit $num_iter: ", result_change)
    end
    time2 = Libc.time() - time0
end

println(result)

fit2_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(problem_definition, result.minimizer)

###########################################################################################
# Refitting GP with full planet signal at found period subtracted (K,P,M0,e,ω,γ-nonlinear)#
###########################################################################################

f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)
h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=current_y)

begin
    time0 = Libc.time()
    global current_hyper = GLOM.remove_zeros(fit2_total_hyperparameters)

    global current_ks = kep_signal(current_ks.K, current_ks.P, current_ks.M0, current_ks.e, current_ks.ω, current_ks.γ)
    println("before full fit: ", kep_parms_str(current_ks))
    # det(∇∇nlogL_kep(problem_definition.y_obs, problem_definition.time, workspace.Σ_obs, current_ks; data_unit=problem_definition.rv_unit*problem_definition.normals[1]))

    results = [Inf, Inf]
    result_change = Inf
    global num_iter = 0
    while result_change > 1e-4 && num_iter < 30
        global current_ks = fit_kepler(problem_definition_rv, workspace.Σ_obs, current_ks)
        println(kep_parms_str(current_ks))
        global current_y = remove_kepler(problem_definition_rv, current_ks)
        fit_GP!(current_hyper)
        results[:] = [results[2], copy(result.minimum)]
        thing = results[1] - results[2]
        if thing < 0; @warn "result increased occured on iteration $num_iter" end
        global result_change = abs(thing)
        global num_iter += 1
        println("change on joint fit $num_iter: ", result_change)
    end
    time3 = Libc.time() - time0
end

println(result)

fit3_total_hyperparameters = GLOM.reconstruct_total_hyperparameters(problem_definition, result.minimizer)
full_ks = kep_signal(current_ks.K, current_ks.P, current_ks.M0, current_ks.e, current_ks.ω, current_ks.γ)

###################
# Post planet fit #
###################

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("kepler hyperparameters")
println(fit3_total_hyperparameters)
fit_nlogL2 = GLOM.nlogL_GLOM!(workspace, problem_definition, fit3_total_hyperparameters; y_obs=current_y)
uE2 = -fit_nlogL2 - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit3_total_hyperparameters, 0) + logprior_kepler(full_ks; use_hk=true)
println(uE2, "\n")

println("best fit keplerian")
println(kep_parms_str(full_ks))

##################################################################################
# refitting noise model to see if a better model was found during planet fitting #
##################################################################################

f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)
g!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)
h!_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, non_zero_hyper)

time1 += fit_GP!(GLOM.remove_zeros(fit3_total_hyperparameters))
println(result)

fit1_total_hyperparameters_temp = GLOM.reconstruct_total_hyperparameters(problem_definition, result.minimizer)

println("first fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("fit after planet hyperparameters")
println(fit1_total_hyperparameters_temp)
fit_nlogL1_temp = GLOM.nlogL_GLOM!(workspace, problem_definition, fit1_total_hyperparameters_temp)
uE1_temp = -fit_nlogL1_temp - nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, fit1_total_hyperparameters_temp, 0)
println(uE1_temp, "\n")

if uE1_temp > uE1
    println("new fit is better, switching hps")
    fit1_total_hyperparameters = fit1_total_hyperparameters_temp
    fit_nlogL1 = fit_nlogL1_temp
    uE1 = uE1_temp
end

##########################
# Evidence approximation #
##########################

# no planet
H1 = (GLOM.∇∇nlogL_GLOM!(workspace, problem_definition, fit1_total_hyperparameters)
    + nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, GLOM.remove_zeros(fit1_total_hyperparameters), 2))
try
    global E1 = GLOM.log_laplace_approximation(H1, -uE1, 0)
catch
    println("Laplace approximation failed for initial GP fit")
    println("det(H1): $(det(H1)) (should've been positive)")
    global E1 = 0
end

# planet
H2 = Matrix(∇∇nlogL_GLOM_and_planet!(workspace, problem_definition_rv, fit3_total_hyperparameters, full_ks; include_kepler_priors=true))
n_hyper = length(GLOM.remove_zeros(fit3_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += nlogprior_kernel_hyperparameters(problem_definition.n_kern_hyper, GLOM.remove_zeros(fit3_total_hyperparameters), 2)
# H2[n_hyper+1:n_hyper+n_kep_parms, n_hyper+1:n_hyper+n_kep_parms] -= logprior_kepler_tot(full_ks; d_tot=2, use_hk=true)
try
    global E2 = log_laplace_approximation(Symmetric(H2), -uE2, 0)
catch
    println("Laplace approximation failed for planet fit")
    println("det(H2): $(det(H2)) (should've been positive)")
    global E2 = 0
end

println("\nlog likelihood for GLOM model: " * string(-fit_nlogL1))
println("log likelihood for GLOM + planet model: " * string(-fit_nlogL2))

println("\nunnormalized posterior for GLOM model: " * string(uE1))
println("unnormalized posterior for GLOM + planet model: " * string(uE2))

println("\nevidence for GLOM model: " * string(E1))
println("evidence for GLOM + planet model: " * string(E2))

saved_likelihoods = [-fit_nlogL1, uE1, E1, -fit_nlogL2, uE2, E2]
save_nlogLs([time1, time2 + time3] ./ 3600, saved_likelihoods, append!(copy(fit1_total_hyperparameters), fit3_total_hyperparameters), full_ks, results_dir)

plot_stuff = !called_from_terminal || E1==0 || E2==0 || !isapprox(original_ks.P, full_ks.P, rtol=1e-1)
if plot_stuff
    include("src/GP_plotting_functions.jl")
    function periodogram_plot(vals::Vector{T} where T<:Real; likelihoods::Bool=true, zoom::Bool=false)
        fig, ax = init_plot()
        if zoom
            inds = (minimum([original_ks.P, best_period]) / 1.5).<period_grid.<(1.5 * maximum([original_ks.P, best_period]))
            fig = plot(ustrip.(period_grid[inds]), vals[inds], color="black")
        else
            fig = plot(ustrip.(period_grid), vals, color="black")
        end
        xscale("log")
        ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        xlabel("Periods (days)")
        if likelihoods
            ylabel("GP log likelihoods")
            axhline(y=-fit_nlogL1, color="k")
            ylim(-fit_nlogL1 - 3, maximum(vals) + 3)
        else
            ylabel("GP log unnormalized posteriors")
            axhline(y=uE1, color="k")
        end
        axvline(x=convert_and_strip_units(u"d", best_period), color="red", linestyle="--")
        file_name_add = ""
        if !likelihoods; file_name_add *= "_ev" end
        if zoom; file_name_add *= "_zoom" end
        save_PyPlot_fig(results_dir * "periodogram" * file_name_add * ".png")
    end

    periodogram_plot(likelihoods; likelihoods=true, zoom=false)
    periodogram_plot(unnorm_evidences; likelihoods=false, zoom=false)

    global hp_string = ""
    for i in 1:problem_definition.n_kern_hyper
        global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit1_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
    end
    GLOM_line_plots(problem_definition_rv, fit1_total_hyperparameters, results_dir * "fit"; hyper_param_string=hp_string)  # , plot_Σ=true, plot_Σ_profile=true)
    global hp_string = ""
    for i in 1:problem_definition.n_kern_hyper
        global hp_string = hp_string * possible_labels[kernel_choice][i] * ": $(round(fit3_total_hyperparameters[end-problem_definition.n_kern_hyper+i], digits=3))  "
    end
    GLOM_line_plots(problem_definition_rv, fit3_total_hyperparameters, results_dir * "fit_full"; fit_ks=full_ks, hyper_param_string=hp_string)


    if (num_kernel_hyperparameters > 0) && called_from_terminal

        actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{12}", L"a_{32}", L"a_{23}"], possible_labels[kernel_choice])
        #I f all 3x3 a's active:  actual_labels = append!([L"a_{11}", L"a_{21}", L"a_{31}", L"a_{12}", L"a_{22}", L"a_{23}", L"a_{31}", L"a_{32}", L"a_{33}"], possible_labels[kernel_choice])

        corner_plot(f_no_print, GLOM.remove_zeros(fit1_total_hyperparameters), results_dir * "corner.png"; input_labels=actual_labels)

        y_obs = remove_kepler(problem_definition_rv, full_ks)
        f_no_print_helper(non_zero_hyper::Vector{T} where T<:Real) = GLOM.nlogL_GLOM!(workspace, problem_definition, non_zero_hyper; y_obs=y_obs)
        corner_plot(f_no_print, GLOM.remove_zeros(fit3_total_hyperparameters), results_dir * "corner_planet.png"; input_labels=actual_labels)

    end

else

    println("not saving figures")

end
