using Pkg
Pkg.activate("examples")
Pkg.instantiate()

# for det()
using LinearAlgebra

# for importing the data from CSV
using DataFrames
using CSV

# For GLOM
import GPLinearODEMaker; GLOM = GPLinearODEMaker

# For this module
using GLOM_RV_Example; GLOM_RV = GLOM_RV_Example

# For units in orbit fitting functions
using UnitfulAstro, Unitful

# For getting times from HDF5 file the SAFE stats were generated from
using HDF5

#####################################################################
# CTRL+F "CHANGE"TO FIND PLACES WHERE YOU SHOULD MAKE MODIFICATIONS #
#####################################################################

## Problem setup

# CHANGE: choose a kernel, I suggest 3 for Matern 5/2 or 4 for Quasi-periodic
# kernel
kernel_choice = 3
kernel_names = ["pp", "se", "m52", "qp", "m52_m52", "se_se"]
kernel_name = kernel_names[kernel_choice]
kernel_function, num_kernel_hyperparameters = GLOM.include_kernel(kernel_name)

# CHANGE: the stars rotation rate which is used as the first guess for some GLOM
# hyperparameters and starting point for priors
star_rot_rate = 25.05  # days

# importing SOAP SAFE data
data = CSV.read("timecorssim_SAFE.csv", DataFrame)[1:2:end, :]

# CHANGE: observation times go here
obs_xs = h5open("D:/Christian/Downloads/res-1000-1years_full_id1.h5")["phases"][1:2:366] * star_rot_rate
# taking out the mean observation times makes the optimization easier for models
# with periodic parameters
GLOM_RV.remove_mean!(obs_xs)

# using Plots
# plot(data[!, "b1"]; label="b1")
# plot!(data[!, "t1"]; label="t1")
# savefig("b1_and_t1.png")
# plot(data[!, "b1"] ./ data[!, "t1"]; label="b1/t1")
# savefig("b1_over_t1.png")
# histogram(data[!, "b1"] ./ data[!, "t1"]; label="b1/t1")
# savefig("b1_over_t1_hist.png")

# CHANGE: rvs and their errors go here
obs_rvs = collect(data[!, "b1"])

# inject_ks = GLOM_RV.kep_signal(; K=1.0u"m/s", P=sqrt(2)*5u"d", M0=rand()*2*π, ω_or_k=rand()*2*π, e_or_h=0.1)
inject_ks = GLOM_RV.kep_signal(; K=1.0u"m/s", P=sqrt(2)*5u"d", M0=3.2992691080593275, ω_or_k=4.110936513051912, e_or_h=0.1)
obs_rvs[:] .+= ustrip.(inject_ks.(obs_xs.*u"d"))
obs_rvs_err = data[!, "b1"] ./ data[!, "t1"]

# CHANGE: activity indicators and thier errors go here
# you can actually have as many as you want, but obviously it will take longer
# to fit
obs_indicator1 = data[!, "b2"]
obs_indicator1_err = data[!, "b2"] ./  data[!, "t2"]
obs_indicator2 = data[!, "b3"]
obs_indicator2_err = data[!, "b3"] ./  data[!, "t3"]

# removing means as the GP model assumes zero mean
GLOM_RV.remove_mean!(obs_rvs)
GLOM_RV.remove_mean!(obs_indicator1)
GLOM_RV.remove_mean!(obs_indicator2)

# CHANGE: change these lines if you add more than 2 indicators
# this takes the data and riffles it together so it takes the form
# [rv_1, ind1_1, ind2_1, rv_2, ind1_2, ind2_2, ...]
n_out = 3  # number of indicators + 1
obs_ys = collect(Iterators.flatten(zip(obs_rvs, obs_indicator1, obs_indicator2)))
obs_noise = collect(Iterators.flatten(zip(obs_rvs_err, obs_indicator1_err, obs_indicator2_err)))

# How many differention orders we want in the GLOM model
n_dif = 2 + 1

# CHANGE: consider changing a0 (the GLOM coefficients that are used, see
# commented lines below)
# If all a's active:
problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a0=[[1. 1 1];[1 1 1];[1 1 1]])
# problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a0=[[1. 1 0];[1 0 1];[1 0 1]])

# Makes the std of each output equal to 1, improves fitting stability
# the normalizations are stored in problem_definition.normals
GLOM.normalize_problem_definition!(problem_definition)

# CHANGE: Setting initial fit values
initial_total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
initial_hypers = [[star_rot_rate], [star_rot_rate], [star_rot_rate], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1]]
append!(initial_total_hyperparameters, initial_hypers[kernel_choice])

initial_total_hyperparameters[:] = [0.0023761507506676236, -0.3054296895488315, -0.11593928780248865, 0.8755168285072893, 11.8214828482746, -2.0796862114288186, 27.485528857812508, -30.289234793914247, -5.34066485785403, 16.17994314368127]
## Fitting GLOM Model

# CHANGE: Setting kernel hyperparameter priors and kick function
# kick functions help avoid saddle points
tighten_lengthscale_priors = 3
if kernel_name in ["pp", "se", "m52"]
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_1λ(hps, d, star_rot_rate, star_rot_rate / 2 / tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_1λ!(hps)
elseif kernel_name == "qp"
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_qp(hps, d, [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate / 2, star_rot_rate / 2, 0.4] ./ tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_qp!(hps)
elseif kernel_name in ["se_se", "m52_m52"]
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_2λ(hps, d, [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate / 2, star_rot_rate / 2, 1] ./ tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_2λ!(hps)
else
    # kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) = custom function
end

fit1_total_hyperparameters, result = GLOM_RV.fit_GLOM(problem_definition, initial_total_hyperparameters, kernel_hyper_priors, add_kick!)
# fit_GLOM returns a vector of num_kernel_hyperparameters gp hyperparameters
# followed by the GLOM coefficients and the Optim result object
workspace = GLOM.nlogL_matrix_workspace(problem_definition, fit1_total_hyperparameters)

## Plotting initial results

plot_xs = collect(LinRange(obs_xs[1]-10, obs_xs[end]+10, 300))
post, post_err, post_obs, post_obs_err = GLOM_RV.GLOM_posteriors(problem_definition, plot_xs, fit1_total_hyperparameters)
GLOM_rvs_at_plot_xs, GLOM_ind1_at_plot_xs, GLOM_ind2_at_plot_xs = post
GLOM_rvs_err_at_plot_xs, GLOM_ind1_err_at_plot_xs, GLOM_ind2_err_at_plot_xs = post_err
GLOM_rvs_at_obs_xs, GLOM_ind1_at_obs_xs, GLOM_ind2_at_obs_xs = post_obs
GLOM_rvs_err_at_obs_xs, GLOM_ind1_err_at_obs_xs, GLOM_ind2_err_at_obs_xs = post_obs_err

activity_rvs = GLOM_rvs_at_obs_xs  # the best guess for activity RVs
clean_rvs = obs_rvs - activity_rvs  # the best guess for RVs without activity

using Plots
plt = scatter(obs_xs, obs_rvs, yerror=obs_rvs_err)
plot!(plt, plot_xs, GLOM_rvs_at_plot_xs, ribbons=GLOM_rvs_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator1, yerror=obs_indicator1_err)
plot!(plt, plot_xs, GLOM_ind1_at_plot_xs, ribbons=GLOM_ind1_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator2, yerror=obs_indicator2_err)
plot!(plt, plot_xs, GLOM_ind2_at_plot_xs, ribbons=GLOM_ind2_err_at_plot_xs, fillalpha=0.3)


## Coefficient exploration

#=
# Could use this to iterate through all of the possible combinations of GLOM
# coefficients
possible_a0s = Matrix[]
GLOM_RV.valid_a0s!(possible_a0s, zeros(n_out, n_dif))
append!(possible_a0s, [ones(n_out, n_dif)])
reverse!(possible_a0s)  # the expensive, filled out possibilities go first
nℓs = Float64[]
all_fit_total_hyperparameters = Vector[]
problem_definitions = GLOM.GLO[]
t0 = Libc.time()
for i in 1:length(possible_a0s)
    a0 = possible_a0s[i]
    problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a0=a0)
    GLOM.normalize_problem_definition!(problem_definition)
    initial_total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
    append!(initial_total_hyperparameters, initial_hypers[kernel_choice])
    fit_total_hyperparameters, result = GLOM_RV.fit_GLOM(problem_definition, initial_total_hyperparameters, kernel_hyper_priors, add_kick!; print_stuff=false)
    append!(problem_definitions, [problem_definition])
    append!(nℓs, result.minimum)
    append!(all_fit_total_hyperparameters, [fit_total_hyperparameters])
    println("\nDone with $(round(100 * i / length(possible_a0s); digits=2))% of a0 possibilities")
    t = Libc.time() - t0
    println("t: $(Int(round(t)))s t_left?: $(round(t * (length(possible_a0s) - i) / i / 60; digits=1)) mins")
end

best_fits = sortperm(nℓs)

plot_xs = collect(LinRange(obs_xs[1]-10, obs_xs[end]+10, 300))
post, post_err, post_obs, post_obs_err = GLOM_RV.GLOM_posteriors(problem_definitions[best_fits[1]], plot_xs, all_fit_total_hyperparameters[best_fits[1]])
GLOM_rvs_at_plot_xs, GLOM_ind1_at_plot_xs, GLOM_ind2_at_plot_xs = post
GLOM_rvs_err_at_plot_xs, GLOM_ind1_err_at_plot_xs, GLOM_ind2_err_at_plot_xs = post_err
GLOM_rvs_at_obs_xs, GLOM_ind1_at_obs_xs, GLOM_ind2_at_obs_xs = post_obs
GLOM_rvs_err_at_obs_xs, GLOM_ind1_err_at_obs_xs, GLOM_ind2_err_at_obs_xs = post_obs_err

using Plots
plt = scatter(obs_xs, obs_rvs, yerror=obs_rvs_err)
plot!(plt, plot_xs, GLOM_rvs_at_plot_xs, ribbons=GLOM_rvs_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator1, yerror=obs_indicator1_err)
plot!(plt, plot_xs, GLOM_ind1_at_plot_xs, ribbons=GLOM_ind1_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator2, yerror=obs_indicator2_err)
plot!(plt, plot_xs, GLOM_ind2_at_plot_xs, ribbons=GLOM_ind2_err_at_plot_xs, fillalpha=0.3)
=#

## Finding a planet?

nlogprior_hyperparameters(total_hyper::Vector, d::Int) = GLOM_RV.nlogprior_hyperparameters(kernel_hyper_priors, problem_definition.n_kern_hyper, total_hyper, d)
problem_definition_rv = GLO_RV(problem_definition, 1u"d", problem_definition.normals[1]u"m/s")

############
# Post fit #
############

println("starting hyperparameters")
println(initial_total_hyperparameters)
initial_nlogL = GLOM.nlogL_GLOM(problem_definition, initial_total_hyperparameters)
initial_uE = -initial_nlogL - nlogprior_hyperparameters(initial_total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nlogL1 = GLOM.nlogL_GLOM(problem_definition, fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - nlogprior_hyperparameters(fit1_total_hyperparameters, 0)
println(uE1, "\n")

#########################
# Keplerian periodogram #
#########################

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
freq_grid = GLOM_RV.autofrequency(problem_definition_rv.time; samples_per_peak=11)
period_grid = 1 ./ reverse(freq_grid)
amount_of_periods = length(period_grid)

Σ_obs = GLOM.Σ_observations(problem_definition, fit1_total_hyperparameters)

# making necessary variables local to all workers
fit1_total_hyperparameters_nz = GLOM.remove_zeros(fit1_total_hyperparameters)
nlogprior_kernel = nlogprior_hyperparameters(fit1_total_hyperparameters_nz, 0)

using Distributed

# CHANGE: can change full_fit to false to just do the epicyclic fit which is
# ~40x faster. Distributing the compute only gives me a factor of 2, so only
# try if the keplerian fitting takes a long time
use_distributed = true
full_fit = true

# concurrency is weird so you may have to run this twice
if use_distributed
    GLOM.auto_addprocs()
    @everywhere using Pkg; @everywhere Pkg.activate("examples"); @everywhere Pkg.instantiate()
    @everywhere using GLOM_RV_Example; @everywhere GLOM_RV = GLOM_RV_Example
    @everywhere import GPLinearODEMaker; @everywhere GLOM = GPLinearODEMaker
    @everywhere using UnitfulAstro, Unitful

    GLOM.sendto(workers(), kernel_name=kernel_name)
    @everywhere GLOM.include_kernel(kernel_name)

    GLOM.sendto(workers(), problem_definition_rv=problem_definition_rv, fit1_total_hyperparameters=fit1_total_hyperparameters, Σ_obs=Σ_obs, nlogprior_kernel=nlogprior_kernel, full_fit=full_fit)
end

@everywhere function fit_kep_hold_P(P::Unitful.Time; fast::Bool=false, kwargs...)
    #initialize with fast epicyclic fit
    ks = GLOM_RV.fit_kepler(problem_definition_rv, Σ_obs, GLOM_RV.kep_signal_epicyclic(P=P))
    if !fast
        ks = GLOM_RV.fit_kepler(problem_definition_rv, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, ks.M0, minimum([ks.e, 0.3]), 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
        return ks
    end
    if ks == nothing
        ks = GLOM_RV.fit_kepler(problem_definition_rv, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, 2 * π * rand(), 0.1, 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
        return ks
    end
    return ks
end
@everywhere function kep_unnormalized_posterior_distributed(P::Unitful.Time; kwargs...)
    ks = fit_kep_hold_P(P; kwargs...)
    if ks == nothing
        return [-Inf, -Inf]
    else
        val = GLOM.nlogL_GLOM(
            problem_definition_rv.GLO,
            fit1_total_hyperparameters;
            Σ_obs=Σ_obs,
            y_obs=GLOM_RV.remove_kepler(problem_definition_rv, ks))
        return [-val, GLOM_RV.logprior_kepler(ks; use_hk=false) - nlogprior_kernel - val]
    end
end
@everywhere kep_unnormalized_posterior_distributed(P::Unitful.Time) = kep_unnormalized_posterior_distributed(P; fast=!full_fit)

likelihoods = zeros(amount_of_periods)
unnorm_posteriors = zeros(amount_of_periods)

@time if use_distributed
    # takes around 115s for 1/2 of SAFE data with 6 workers and 5000 periods
    holder = pmap(x->kep_unnormalized_posterior_distributed(x), period_grid, batch_size=Int(floor(amount_of_periods / (nworkers() + 1)) + 1))
    likelihoods[:] = [holder[i][1] for i in 1:length(holder)]
    unnorm_posteriors[:] = [holder[i][2] for i in 1:length(holder)]
else
    # takes around 15s for 101501 data and 1200 periods
    for i in 1:amount_of_periods
        likelihoods[i], unnorm_posteriors[i] = kep_unnormalized_posterior_distributed(period_grid[i])
    end
end

best_periods = period_grid[GLOM_RV.find_modes(unnorm_posteriors; amount=10)]
best_period = best_periods[1]

println("found period:    $(ustrip(best_period)) days")

# using Plots
# plot(ustrip.(period_grid), likelihoods; xaxis=:log, leg=false)
# plot(ustrip.(period_grid), unnorm_posteriors; xaxis=:log, leg=false)

####################################################################################################
# Refitting GP with full planet signal at found period subtracted (K,ω,γ-linear, P,M0,e-nonlinear) #
####################################################################################################
remainder(vec, x) = [i > 0 ? i % x : (i % x) + x for i in vec]

current_ks = fit_kep_hold_P(best_period; print_stuff=true)
println("before wright fit: ", GLOM_RV.kep_parms_str(current_ks))

#=
plot_kep_xs = collect(LinRange(0, ustrip(best_period), 1000))
# scatter(remainder(problem_definition.x_obs, ustrip(best_period)), ustrip.(problem_definition_rv.rv); yerror=ustrip.(problem_definition_rv.rv_noise), label="data")
scatter(remainder(problem_definition.x_obs, ustrip(best_period)), clean_rvs; yerror=GLOM_rvs_err_at_obs_xs, label="\"clean\" data")
plot!(plot_kep_xs, ustrip.(current_ks.(plot_kep_xs.*u"d")); label="kep")
=#
fit2_total_hyperparameters, current_ks = GLOM_RV.fit_GLOM_and_kep!(workspace,
    problem_definition_rv, fit1_total_hyperparameters, kernel_hyper_priors,
    add_kick!, current_ks; avoid_saddle=false)

# # these should be near 0
# GLOM_RV.test_∇nlogL_kep(problem_definition_rv, workspace.Σ_obs, current_ks; include_priors=true)
###########################################################################################
# Refitting GP with full planet signal at found period subtracted (K,P,M0,e,ω,γ-nonlinear)#
###########################################################################################

current_ks = GLOM_RV.kep_signal(current_ks)
println("\nbefore full fit: ", GLOM_RV.kep_parms_str(current_ks))
fit3_total_hyperparameters, current_ks = GLOM_RV.fit_GLOM_and_kep!(workspace,
    problem_definition_rv, fit2_total_hyperparameters, kernel_hyper_priors,
    add_kick!, current_ks; avoid_saddle=true)

full_ks = GLOM_RV.kep_signal(current_ks)

# # these should be near 0
# GLOM_RV.test_∇nlogL_kep(problem_definition_rv, workspace.Σ_obs, full_ks; include_priors=true)

###################
# Post planet fit #
###################

println("fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("kepler hyperparameters")
println(fit3_total_hyperparameters)
fit_nlogL2 = GLOM.nlogL_GLOM!(workspace, problem_definition, fit3_total_hyperparameters; y_obs=GLOM_RV.remove_kepler(problem_definition_rv, full_ks))
uE2 = -fit_nlogL2 - nlogprior_hyperparameters(fit3_total_hyperparameters, 0) + GLOM_RV.logprior_kepler(full_ks; use_hk=true)
println(uE2, "\n")

println("best fit keplerian")
println(GLOM_RV.kep_parms_str(full_ks))

##################################################################################
# refitting noise model to see if a better model was found during planet fitting #
##################################################################################

fit1_total_hyperparameters_temp, result = GLOM_RV.fit_GLOM(
    problem_definition,
    fit3_total_hyperparameters,
    kernel_hyper_priors,
    add_kick!)

println(result)

println("first fit hyperparameters")
println(fit1_total_hyperparameters)
println(uE1, "\n")

println("fit after planet hyperparameters")
println(fit1_total_hyperparameters_temp)
fit_nlogL1_temp = GLOM.nlogL_GLOM!(workspace, problem_definition, fit1_total_hyperparameters_temp)
uE1_temp = -fit_nlogL1_temp - nlogprior_hyperparameters(fit1_total_hyperparameters_temp, 0)
println(uE1_temp, "\n")

if uE1_temp > uE1
    println("new fit is better, switching hps")
    fit1_total_hyperparameters[:] = fit1_total_hyperparameters_temp
    fit_nlogL1 = fit_nlogL1_temp
    uE1 = uE1_temp
end

##########################
# Evidence approximation #
##########################

# no planet
H1 = (GLOM.∇∇nlogL_GLOM(problem_definition, fit1_total_hyperparameters)
    + nlogprior_hyperparameters(GLOM.remove_zeros(fit1_total_hyperparameters), 2))
try
    global E1 = GLOM.log_laplace_approximation(H1, -uE1, 0)
catch err
    if isa(err, DomainError)
        println("Laplace approximation failed for initial GP fit")
        println("det(H1): $(det(H1)) (should've been positive)")
        global E1 = 0
    else
        rethrow()
    end
end

# planet
H2 = Matrix(GLOM_RV.∇∇nlogL_GLOM_and_planet!(workspace, problem_definition_rv, fit3_total_hyperparameters, full_ks; include_kepler_priors=true))
n_hyper = length(GLOM.remove_zeros(fit3_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += nlogprior_hyperparameters(GLOM.remove_zeros(fit3_total_hyperparameters), 2)
try
    global E2 = GLOM.log_laplace_approximation(Symmetric(H2), -uE2, 0)
catch err
    if isa(err, DomainError)
        println("Laplace approximation failed for planet fit")
        println("det(H2): $(det(H2)) (should've been positive)")
        global E2 = 0
    else
        rethrow()
    end
end

println("\nlog likelihood for GLOM model: " * string(-fit_nlogL1))
println("log likelihood for GLOM + planet model: " * string(-fit_nlogL2))

println("\nunnormalized posterior for GLOM model: " * string(uE1))
println("unnormalized posterior for GLOM + planet model: " * string(uE2))

println("\nevidence for GLOM model: " * string(E1))
println("evidence for GLOM + planet model: " * string(E2))

# # This should be pretty close to true
# GLOM_RV.test_∇∇nlogL_kep(problem_definition_rv, workspace.Σ_obs, current_ks; include_priors=true)

## Plotting final results

post, post_err, post_obs, post_obs_err = GLOM_RV.GLOM_posteriors(problem_definition, plot_xs, fit3_total_hyperparameters; y_obs=GLOM_RV.remove_kepler(problem_definition_rv, full_ks))
GLOM_rvs_at_plot_xs, GLOM_ind1_at_plot_xs, GLOM_ind2_at_plot_xs = post
GLOM_rvs_err_at_plot_xs, GLOM_ind1_err_at_plot_xs, GLOM_ind2_err_at_plot_xs = post_err
GLOM_rvs_at_obs_xs, GLOM_ind1_at_obs_xs, GLOM_ind2_at_obs_xs = post_obs
GLOM_rvs_err_at_obs_xs, GLOM_ind1_err_at_obs_xs, GLOM_ind2_err_at_obs_xs = post_obs_err

activity_rvs = GLOM_rvs_at_obs_xs  # the best guess for activity RVs
clean_rvs = obs_rvs - activity_rvs  # the best guess for RVs without activity

using Plots
plt = scatter(obs_xs, GLOM_RV.remove_kepler(problem_definition_rv, full_ks)[1:3:end], yerror=obs_rvs_err)
plot!(plt, plot_xs, GLOM_rvs_at_plot_xs, ribbons=GLOM_rvs_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator1, yerror=obs_indicator1_err)
plot!(plt, plot_xs, GLOM_ind1_at_plot_xs, ribbons=GLOM_ind1_err_at_plot_xs, fillalpha=0.3)

plt = scatter(obs_xs, obs_indicator2, yerror=obs_indicator2_err)
plot!(plt, plot_xs, GLOM_ind2_at_plot_xs, ribbons=GLOM_ind2_err_at_plot_xs, fillalpha=0.3)

println(fit3_total_hyperparameters)
