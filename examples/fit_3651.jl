using Pkg
Pkg.activate("examples")
Pkg.instantiate()

# for std()
using Statistics

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

using Plots
using JLD2

#####################################################################
# CTRL+F "CHANGE"TO FIND PLACES WHERE YOU SHOULD MAKE MODIFICATIONS #
#####################################################################

## Problem setup
star_str = "HD_3651_neid"

# importing data
data_dir = "NEID/glom/"

@load "jld2/reduced_3651.jld2" times_nu rvs_red_aug rvs_σ_red_aug rvs_red_greedy_aug rvs_σ_red_greedy_aug neid_time neid_rv neid_rv_σ
include("data_locs.jl")
star = "3651"
pipeline_path = neid_save_path * star * "/neid_pipeline.jld2"
@load pipeline_path d_act_tot d_lcs

# CHANGE: the stars rotation rate which is used as the first guess for some GLOM
# hyperparameters and starting point for priors
# initial_hypers = [0.0069, 82.6, 1] # days
# initial_hypers = [82., 200, 1] # days
initial_hypers = [44.5] # days
# initial_hypers = [82., 200, 1] # days

# CHANGE: choose a kernel, I suggest 3 for Matern 5/2 or 4 for Quasi-periodic
# kernel
# kernel_name = "m52_m52"
# kernel_name = "qp"
kernel_name = "m52"
# kernel_name = "mp"
kernel_function, num_kernel_hyperparameters = GLOM.include_kernel(kernel_name)

obs_xs = neid_time
# obs_xs = neid_time
sort_inds = sortperm(obs_xs)
obs_xs = obs_xs[sort_inds]

# taking out the mean observation times makes the optimization easier for models
# with periodic parameters
GLOM_RV.remove_mean!(obs_xs)

# CHANGE: Where to save intermediate data products
save_dir = data_dir * "data/" * star_str * "_"
fig_dir = data_dir * "figs/" * star_str * "_"

# CHANGE: rvs and their errors go here
obs_rvs = neid_rv[sort_inds]
obs_rvs_err = neid_rv_σ[sort_inds]
GLOM_RV.remove_mean!(obs_rvs)

# CHANGE: whether to use saved values if they exist
use_saved = false

## Adding indicators

# initializing RV and indicator array
rvs_and_inds_holder = [obs_rvs]
rvs_and_inds_err_holder = [obs_rvs_err]

function append_vec!(to_be_append_to::Vector{Vector{T}}, to_append::Vector{T}; remove_mean::Bool=true) where T
    if remove_mean
        GLOM_RV.remove_mean!(to_append)
    end
    append!(to_be_append_to, [to_append])
end
function add_indicator!(rvs_and_inds::Vector{Vector{T}}, rvs_and_inds_err::Vector{Vector{T}}, indicator::Vector{T}, indicator_err::Vector{T}) where T
    append_vec!(rvs_and_inds, indicator; remove_mean=true)
    append_vec!(rvs_and_inds_err, indicator_err; remove_mean=false)
end

# CHANGE: activity indicators and their errors go here
# you can actually have as many as you want, but obviously it will take longer
# to fit
add_indicator!(rvs_and_inds_holder, rvs_and_inds_err_holder, d_act_tot["CaIIHK"][sort_inds], d_act_tot["CaIIHK_σ"][sort_inds])

# CHANGE: which of the indicators do you want use?
inds_to_use = 1:length(rvs_and_inds_holder)
inds_to_use = 1:length(rvs_and_inds_holder)
@assert inds_to_use[1] == 1  # inds_to_use needs to include the RVs first

rvs_and_inds = rvs_and_inds_holder[inds_to_use]
rvs_and_inds_err = rvs_and_inds_err_holder[inds_to_use]
n_out = length(rvs_and_inds)  # number of indicators + 1
obs_ys = collect(Iterators.flatten(zip(rvs_and_inds...)))
obs_noise = collect(Iterators.flatten(zip(rvs_and_inds_err...)))

# How many differention orders we want in the GLOM model
n_dif = 2 + 1

# CHANGE: consider changing a (the GLOM coefficients that are used, see
# commented lines below)
# If all a's active:
# glo = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a=ones(n_out, n_dif))
glo = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a=[[1. 1 0];[1 0 1]])

# Makes the std of each output equal to 1, improves fitting stability
# the normalizations are stored in glo.normals
GLOM.normalize_GLO!(glo)

# CHANGE: Setting initial fit values
initial_total_hyperparameters = collect(Iterators.flatten(glo.a))
append!(initial_total_hyperparameters, initial_hypers)

## Fitting GLOM Model

# CHANGE: Setting kernel hyperparameter priors and kick function
# kick functions help avoid saddle points
tighten_lengthscale_priors = 1
if kernel_name in ["pp", "se", "m52"]
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_1λ(hps, d, initial_hypers, initial_hypers ./ tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_1λ!(hps)
elseif kernel_name in ["qp", "mp"]
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_qp(hps, d, initial_hypers, initial_hypers ./ tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_qp!(hps)
elseif kernel_name in ["se_se", "m52_m52"]
    kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) =
        GLOM_RV.kernel_hyper_priors_2λ(hps, d, initial_hypers, initial_hypers ./ tighten_lengthscale_priors)
    add_kick!(hps::Vector{<:Real}) = GLOM_RV.add_kick_2λ!(hps)
else
    # kernel_hyper_priors(hps::Vector{<:Real}, d::Integer) = custom function
end

if use_saved && isfile(save_dir*"fit1.jld2")
    # GLOM.include_kernel(kernel_name)
    @load save_dir*"fit1.jld2" glo fit1_total_hyperparameters
else
    # fit_GLOM returns a vector of num_kernel_hyperparameters gp hyperparameters
    # followed by the GLOM coefficients and the Optim result object
    fit1_total_hyperparameters, result = GLOM_RV.fit_GLOM(glo, initial_total_hyperparameters, kernel_hyper_priors, add_kick!)
    println(result)

    @save save_dir*"fit1.jld2" glo fit1_total_hyperparameters
end

## Plotting initial results
# plot_xs = sort(append!(collect(LinRange(-169.73, -169.69, 300)), obs_xs))
# GLOM_at_plot_xs, GLOM_err_at_plot_xs, GLOM_at_obs_xs = GLOM_RV.GLOM_posteriors(glo, plot_xs, fit1_total_hyperparameters)
# scatter(obs_xs, obs_rvs; xrange=(-169.73, -169.69))
# plot!(plot_xs, GLOM_at_plot_xs[1]; ribbons=GLOM_err_at_plot_xs)

plot_xs = sort(append!(collect((obs_xs[1] - 10):3:(obs_xs[end] + 10)), append!([(obs_xs[i] + obs_xs[i - 1]) / 2 for i in 2:length(obs_xs)], obs_xs)))



include(pathof(GLOM_RV) * "../../../examples/plot_helpers.jl")
plot_helper(prefix::String, fit_total_hyperparameters::AbstractVector) =
    plot_helper(glo, plot_xs, rvs_and_inds, rvs_and_inds_err, prefix, fit_total_hyperparameters)
plot_helper(prefix::String, ks::GLOM_RV.KeplerSignal, fit_total_hyperparameters::AbstractVector) =
    plot_helper(glo_rv, plot_xs, rvs_and_inds, rvs_and_inds_err, prefix, ks, fit_total_hyperparameters)

plot_helper("fit1_", fit1_total_hyperparameters)

## Finding a planet?

# http://www.openexoplanetcatalogue.com/planet/Proxima%20Centauri%20d/

nlogprior_hyperparameters(total_hyper::Vector, d::Int) = GLOM_RV.nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, total_hyper, d)
glo_rv = GLO_RV(glo, 1u"d", glo.normals[1]u"m/s")

############
# Post fit #
############

println("starting hyperparameters")
println(initial_total_hyperparameters)
initial_nlogL = GLOM.nlogL_GLOM(glo, initial_total_hyperparameters)
initial_uE = -initial_nlogL - nlogprior_hyperparameters(initial_total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
fit_nlogL1 = GLOM.nlogL_GLOM(glo, fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - nlogprior_hyperparameters(fit1_total_hyperparameters, 0)
println(uE1, "\n")

#########################
# Keplerian periodogram #
#########################

# sample linearly in frequency space so that we get periods from the 1 / uneven Nyquist
freq_grid = GLOM_RV.autofrequency(glo_rv.time; samples_per_peak=11)
period_grid = 1 ./ reverse(freq_grid)
period_grid = period_grid[searchsortedfirst(ustrip.(period_grid), 50):searchsortedfirst(ustrip.(period_grid), 70)]
amount_of_periods = length(period_grid)

Σ_obs = GLOM.Σ_observations(glo, fit1_total_hyperparameters)

# making necessary variables local to all workers
fit1_total_hyperparameters_nz = GLOM.remove_zeros(fit1_total_hyperparameters)
nlogprior_kernel = nlogprior_hyperparameters(fit1_total_hyperparameters_nz, 0)

using Distributed

# CHANGE: can change full_fit to false to just do the epicyclic fit which is
# ~40x faster. Distributing the compute only gives me a factor of 2, so only
# try if the keplerian fitting takes a long time
use_distributed = false
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

    GLOM.sendto(workers(), glo_rv=glo_rv, fit1_total_hyperparameters=fit1_total_hyperparameters, Σ_obs=Σ_obs, nlogprior_kernel=nlogprior_kernel, full_fit=full_fit)
end

@everywhere function fit_kep_hold_P(P::Unitful.Time; fast::Bool=false, kwargs...)
    #initialize with fast epicyclic fit
    ks = GLOM_RV.fit_kepler(glo_rv, Σ_obs, GLOM_RV.kep_signal_epicyclic(P=P))
    if !fast
        ks = GLOM_RV.fit_kepler(glo_rv, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, ks.M0, minimum([ks.e, 0.3]), 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
        return ks
    end
    if ks == nothing
        ks = GLOM_RV.fit_kepler(glo_rv, Σ_obs, GLOM_RV.kep_signal_wright(0u"m/s", P, 2 * π * rand(), 0.1, 0, 0u"m/s"); hold_P=true, avoid_saddle=false, print_stuff=false, kwargs...)
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
            glo_rv.GLO,
            fit1_total_hyperparameters;
            Σ_obs=Σ_obs,
            y_obs=GLOM_RV.remove_kepler(glo_rv, ks))
        return [-val, GLOM_RV.logprior_kepler(ks; use_hk=false) - nlogprior_kernel - val]
    end
end
@everywhere kep_unnormalized_posterior_distributed(P::Unitful.Time) = kep_unnormalized_posterior_distributed(P; fast=!full_fit)


if use_saved && isfile(save_dir*"period.jld2")
    # need to using Unitful before this will work
    @load save_dir*"period.jld2" likelihoods unnorm_posteriors period_grid best_period
else
    likelihoods = zeros(amount_of_periods)
    unnorm_posteriors = zeros(amount_of_periods)

    @time if use_distributed
        # takes around a minute for 26965 data and 3000 periods
        holder = pmap(x->kep_unnormalized_posterior_distributed(x), period_grid, batch_size=Int(floor(amount_of_periods / (nworkers() + 1)) + 1))
        likelihoods[:] = [holder[i][1] for i in 1:length(holder)]
        unnorm_posteriors[:] = [holder[i][2] for i in 1:length(holder)]
    else
        # takes around minutes for 101501 data and 3000 periods
        for i in 1:amount_of_periods
            likelihoods[i], unnorm_posteriors[i] = kep_unnormalized_posterior_distributed(period_grid[i])
        end
    end

    best_periods = period_grid[GLOM_RV.find_modes(unnorm_posteriors; amount=3)]
    best_period = best_periods[1]

    println("found period:    $(ustrip(best_period)) days")

    @save save_dir*"period.jld2" likelihoods unnorm_posteriors period_grid best_period best_periods
end

plot(ustrip.(period_grid), likelihoods; xaxis=:log, leg=false, xlabel="Period (days)", ylabel="log(Likelihood)", title = star_str * " Keplerian Periodogram")
png(fig_dir * "period_lik")
plot(ustrip.(period_grid), unnorm_posteriors; xaxis=:log, leg=false, xlabel="Period (days)", ylabel="log(Unnormalized Posterior)", title = star_str * " Keplerian Periodogram")
# vline!([11.1843, 5.122])
png(fig_dir * "period_evi")
# period_thres = 20
# scatter(ustrip.(period_grid[GLOM_RV.find_modes(unnorm_posteriors; amount=50)]), ylim=(0,period_thres), leg=false, ylabel="Period (days)", xlabel="Peak Ranking", title = star_str * " Periodogram Peaks <$period_thres days")
# hline!([11.1843, 5.122])
# png(fig_dir * "period_rankings")

####################################################################################################
# Refitting GP with full planet signal at found period subtracted (K,ω,γ-linear, P,M0,e-nonlinear) #
####################################################################################################
remainder(vec, x) = [i > 0 ? i % x : (i % x) + x for i in vec]

# Need to redefine Σ_obs for fit_kep_hold_P()
Σ_obs = GLOM.Σ_observations(glo, fit1_total_hyperparameters)
current_ks = fit_kep_hold_P(best_period; print_stuff=true)
println("before GLOM+Wright fit: ", GLOM_RV.kep_parms_str(current_ks))

if use_saved && isfile(save_dir*"fit3.jld2")
    @load save_dir*"fit3.jld2" glo fit3_total_hyperparameters current_ks
    glo_rv = GLO_RV(glo, 1u"d", glo.normals[1]u"m/s")
else
    # 400s
    @time fit3_total_hyperparameters, current_ks = GLOM_RV.fit_GLOM_and_kep(glo_rv,
    fit1_total_hyperparameters, kernel_hyper_priors, add_kick!, current_ks;
    avoid_saddle=false, fit_alpha=1e-3)

    @save save_dir*"fit3.jld2" glo fit3_total_hyperparameters current_ks
end

plot_helper("fit3_", current_ks, fit3_total_hyperparameters)
fit4_total_hyperparameters = fit3_total_hyperparameters
full_ks = GLOM_RV.kep_signal(current_ks)

# # these should be near 0
# GLOM_RV.test_∇nlogL_kep(glo_rv, workspace.Σ_obs, current_ks; include_priors=true)
# ###########################################################################################
# # Refitting GP with full planet signal at found period subtracted (K,P,M0,e,ω,γ-nonlinear)#
# ###########################################################################################
#
# if use_saved && isfile(save_dir*"fit4.jld2")
#     @load save_dir*"fit4.jld2" glo fit4_total_hyperparameters current_ks
#     glo_rv = GLO_RV(glo, 1u"d", glo.normals[1]u"m/s")
# else
#     # 200s
#     current_ks = GLOM_RV.kep_signal(current_ks)
#     println("\nbefore full fit: ", GLOM_RV.kep_parms_str(current_ks))
#     @time fit4_total_hyperparameters, current_ks = GLOM_RV.fit_GLOM_and_kep(glo_rv,
#     fit3_total_hyperparameters, kernel_hyper_priors, add_kick!, current_ks;
#     avoid_saddle=true)
#
#     @save save_dir*"fit4.jld2" glo fit4_total_hyperparameters current_ks
# end
#
# plot_helper("fit4_", current_ks, fit4_total_hyperparameters)
#
# full_ks = GLOM_RV.kep_signal(current_ks)
#
# # # these should be near 0
# # GLOM_RV.test_∇nlogL_kep(glo_rv, workspace.Σ_obs, full_ks; include_priors=true)

###################
# Post planet fit #
###################
println("fit hyperparameters")
println(fit1_total_hyperparameters, "\n")
# println(uE1, "\n")

println("kepler hyperparameters")
println(fit4_total_hyperparameters, "\n")
fit_nlogL2 = GLOM.nlogL_GLOM(glo, fit4_total_hyperparameters; y_obs=GLOM_RV.remove_kepler(glo_rv, full_ks))
uE2 = -fit_nlogL2 - nlogprior_hyperparameters(fit4_total_hyperparameters, 0) + GLOM_RV.logprior_kepler(full_ks; use_hk=true)
# println(uE2, "\n")

# println("best fit keplerian")
# println(GLOM_RV.kep_parms_str(full_ks))

# ##################################################################################
# # refitting noise model to see if a better model was found during planet fitting #
# ##################################################################################
#
# fit1_total_hyperparameters_temp, result = GLOM_RV.fit_GLOM(
#     glo,
#     fit4_total_hyperparameters,
#     kernel_hyper_priors,
#     add_kick!)
#
# println(result)
#
# println("first fit hyperparameters")
# println(fit1_total_hyperparameters)
# println(uE1, "\n")
#
# println("fit after planet hyperparameters")
# println(fit1_total_hyperparameters_temp)
# fit_nlogL1_temp = GLOM.nlogL_GLOM(glo, fit1_total_hyperparameters_temp)
# uE1_temp = -fit_nlogL1_temp - nlogprior_hyperparameters(fit1_total_hyperparameters_temp, 0)
# println(uE1_temp, "\n")
#
# if uE1_temp > uE1
#     println("new fit is better, switching hps")
#     fit1_total_hyperparameters[:] = fit1_total_hyperparameters_temp
#     fit_nlogL1 = fit_nlogL1_temp
#     uE1 = uE1_temp
#     @save save_dir*"fit1.jld2" glo fit1_total_hyperparameters
# end

##########################
# Evidence approximation #
##########################

# no planet
H1 = (GLOM.∇∇nlogL_GLOM(glo, fit1_total_hyperparameters)
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
H2 = Matrix(GLOM_RV.∇∇nlogL_GLOM_and_planet(glo_rv, fit4_total_hyperparameters, full_ks; include_kepler_priors=true))
n_hyper = length(GLOM.remove_zeros(fit4_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += nlogprior_hyperparameters(GLOM.remove_zeros(fit4_total_hyperparameters), 2)
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
println("ln(evidence ratio: $(E2-E1)")
fit_total_hyperparameters_σ = sqrt.(diag(inv(H2)))
ks_σ = fit_total_hyperparameters_σ[end-5:end]
println(GLOM_RV.kep_parms_str(full_ks, ks_σ))
