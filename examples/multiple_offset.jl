## This script shows how one could use GLOM to find a planet by analyzing a simulated GLOM model and adding a planet and multiple RV offsets
## Note: I HIGHLY recommend taking out as much of the known offset as possible before doing the final offset twidling with GLOM

using Pkg
Pkg.activate("examples")
Pkg.instantiate()

using Statistics  # for mean()
using LinearAlgebra  # for det()
import GPLinearODEMaker as GLOM
import GLOM_RV_Example as GLOM_RV
using UnitfulAstro, Unitful  # for units in orbit fitting functions
using Plots  # for plotting
using LaTeXStrings

plot_dir = "examples/figs/multiple_offset/"

####################
# Simulation setup #
####################

# Kernel being used. See GLOM.include_kernel for details on kernel availability
kernel_name = "m52"
# what GLOM coefficients are being used
a0 =[[1. 1 0];[1 0 1]]
# Observation times
obs_xs = sort!(append!(LinRange(1,30., 30) .+ 0.1.*rand(30), LinRange(1,30., 30) .+ 0.1.*rand(30)))  # twice nightly observations for a 30 days
# Lengthscale of the input GLOM model
λ = 5.  # days
# Input RV noise
rv_noise = 0.1 .* ones(length(obs_xs))  # 10 cm/s RV noise
# Input indicator noise
indicator_noise = 0.3 .* ones(length(obs_xs))
# Input planet
inject_ks = GLOM_RV.kep_signal(; K=1.0u"m/s", P=sqrt(2)*5u"d", M0=3.2992691080593275, ω_or_k=4.110936513051912, e_or_h=0.1)

# the observations have two different RV offsets
n_obs = length(obs_xs)
mid_obs = Int(round(n_obs/2))
offset_mask = ones(Bool, n_obs, 2)
offset_mask[mid_obs+1:end, 1] .= false
offset_mask[1:mid_obs, 2] .= false
offset = GLOM_RV.Offset(1*[-1., 1.] .* u"m/s", offset_mask)
current_offset = copy(offset)

###################
# Data simulation #
###################

# simulating activity and RVs from the GLOM prior
kernel_function, num_kernel_hyperparameters = GLOM.include_kernel(kernel_name)
initial_hypers = [λ]
initial_total_hyperparameters = collect(Iterators.flatten(a0))
append!(initial_total_hyperparameters, initial_hypers)
Σ = GLOM.prior_covariance(kernel_function, initial_hypers, a0, obs_xs)
n_out, n_dif = size(a0)
original_ys = Σ.L * randn(n_out * n_obs)
rvs, indicator = GLOM.unriffle(original_ys, n_out)

# adding white noise, planet, and offsets
rvs .+= rv_noise .* randn(n_obs) + ustrip.(inject_ks.(obs_xs.*u"d")+ offset()) # add noise, planet, and offsets
indicator .+= indicator_noise .* randn(n_obs)  # add noise and planet

##############
# GLOM Setup #
##############

# creating the GLO object for GLOM
ref_t = mean(obs_xs)
GLOM_RV.remove_mean!(obs_xs)
# GLOM_RV.remove_mean!(rvs)
# GLOM_RV.remove_mean!(indicator)
obs_ys = GLOM.riffle([rvs, indicator])
obs_noise = GLOM.riffle([rv_noise, indicator_noise])
# rvs_and_inds_holder = [obs_rvs]
# rvs_and_inds_err_holder = [obs_rvs_err]
# GLOM_RV.add_indicator!(rvs_and_inds_holder, rvs_and_inds_err_holder, indicator, indicator_noise)
# obs_ys = GLOM.riffle(rvs_and_inds_holder)
# obs_noise = GLOM.riffle(rvs_and_inds_err_holder)

glo = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a=copy(a0))
GLOM.normalize_GLO!(glo)

# defining functions for GP hyperparameter priors and how to kick the fitting out of saddle points (see other kernel_hyper_priors and add_kick functions in fit_GLOM_functions.jl)
tighten_lengthscale_priors = 1
kernel_hyper_priors(hps::AbstractVector{<:Real}, d::Integer) =
    GLOM_RV.kernel_hyper_priors_1λ(hps, d, initial_hypers, initial_hypers ./ tighten_lengthscale_priors)
add_kick!(hps::AbstractVector{<:Real}) = GLOM_RV.add_kick_1λ!(hps)

##################
# Fit GLOM model #
##################

# fitting the GLOM model to the simulated data
fit1_total_hyperparameters, result = GLOM_RV.fit_GLOM(glo, initial_total_hyperparameters, kernel_hyper_priors, add_kick!)

# just a helper function to make it easier to evaluate the priors on the hyperparameters
nlogprior_hyperparameters(total_hyper::Vector, d::Int) = GLOM_RV.nlogprior_hyperparameters(kernel_hyper_priors, glo.n_kern_hyper, total_hyper, d)
fit_nlogL1 = GLOM.nlogL_GLOM(glo, fit1_total_hyperparameters)
H1 = (GLOM.∇∇nlogL_GLOM(glo, fit1_total_hyperparameters)
    + nlogprior_hyperparameters(GLOM.remove_zeros(fit1_total_hyperparameters), 2))
fit1_total_hyperparameters_σ = GLOM_RV.errs_from_hessian(H1)
lengthscale_ind = length(fit1_total_hyperparameters_σ)
title_helper(nzhypers, hypers_σ, ℓ) = L" \lambda_{M^5/_2}=%$(GLOM.rounded(nzhypers[lengthscale_ind])) \pm %$(GLOM.rounded(hypers_σ[lengthscale_ind])), \ \ell=%$(GLOM.rounded(ℓ))"
x_samp = GLOM_RV.plot_points(glo; max_extra=100)
plt = GLOM_RV.make_plot(glo, fit1_total_hyperparameters, 
    [L"\textrm{RVs}", L"\textrm{Indicator}"], 
    [L"\textrm{RV \ (m/s)}", L"\textrm{Indicator}"]; 
    title=title_helper(GLOM.remove_zeros(fit1_total_hyperparameters), fit1_total_hyperparameters_σ, fit_nlogL1), 
    x_samp=x_samp, x_mean=ref_t)
png(plot_dir * "glom")

# creating the RV version of the GLO object
glo_rv = GLOM_RV.GLO_RV(glo, 1u"d", glo.normals[1]u"m/s")

println("starting hyperparameters")
println(initial_total_hyperparameters)
initial_nlogL = GLOM.nlogL_GLOM(glo, initial_total_hyperparameters)
initial_uE = -initial_nlogL - nlogprior_hyperparameters(initial_total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - nlogprior_hyperparameters(fit1_total_hyperparameters, 0)
println(uE1, "\n")

Σ_obs = GLOM.Σ_observations(glo, fit1_total_hyperparameters)

#######################################
# Keplerian periodogram planet search #
#######################################

# search for planet period with a keplerian peridogram
period_grid, likelihoods, unnorm_posteriors, kss, offsets = GLOM_RV.keplerian_periodogram(glo_rv, current_offset, fit1_total_hyperparameters; Σ_obs=Σ_obs, nlogprior_kernel=nlogprior_hyperparameters(GLOM.remove_zeros(fit1_total_hyperparameters), 0))
best_periods = period_grid[GLOM_RV.find_modes(likelihoods)]
best_period = best_periods[1]
# best_period = inject_ks.P
println("found period: $(ustrip(best_period)) days")
current_ks, current_offset = GLOM_RV.fit_kep_hold_P(best_period, glo_rv, current_offset, Σ_obs)
println("before GLOM+Keplerian fit:")
println(GLOM_RV.kep_and_offset_parms_str(current_ks, current_offset))
println(GLOM_RV.kep_and_offset_parms_str(inject_ks, offset))
#TODO make plots here

plt = GLOM_RV.periodogram_plot(period_grid, likelihoods; font_size=14, xaxis=:log, truth=ustrip(inject_ks.P))
png(plot_dir * "periodogram")

###############################
# fit the GLOM + planet model #
###############################

fit2_total_hyperparameters, current_ks = GLOM_RV.fit_GLOM_and_kep(glo_rv,
    fit1_total_hyperparameters, kernel_hyper_priors, add_kick!, current_ks, current_offset;
    avoid_saddle=false, fit_alpha=1e-3)

# convert the semi-linear keplerian model to a fully non-linear one
full_ks = GLOM_RV.kep_signal(current_ks)

fit_nlogL2 = GLOM.nlogL_GLOM(glo, fit2_total_hyperparameters; y_obs=GLOM_RV.remove_kepler(glo_rv, full_ks, current_offset))
uE2 = -fit_nlogL2 - nlogprior_hyperparameters(fit2_total_hyperparameters, 0) + GLOM_RV.logprior_kepler(full_ks; use_hk=true)

H2 = Matrix(GLOM_RV.∇∇nlogL_GLOM_and_kep(glo_rv, fit2_total_hyperparameters, full_ks, current_offset; include_kepler_priors=true))
n_hyper = length(GLOM.remove_zeros(fit2_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += nlogprior_hyperparameters(GLOM.remove_zeros(fit2_total_hyperparameters), 2)
fit2_total_hyperparameters_σ = GLOM_RV.errs_from_hessian(H2)

plt = GLOM_RV.make_plot(glo_rv, full_ks, current_offset, fit2_total_hyperparameters, 
    [L"\textrm{RVs - Keplerian - Offset}", L"\textrm{Indicator}"], 
    [L"\textrm{RV \ (m/s)}", L"\textrm{Indicator}"]; 
    title=title_helper(GLOM.remove_zeros(fit2_total_hyperparameters), fit2_total_hyperparameters_σ, fit_nlogL2), 
    x_samp=x_samp, x_mean=ref_t)
png(plot_dir * "glom_and_kep")

plt, plt_phase = GLOM_RV.keplerian_plot(glo_rv, fit2_total_hyperparameters, full_ks, current_offset);
png(plt, plot_dir * "kep")
png(plt_phase, plot_dir * "kep_phase")


##########################
# Evidence approximation #
##########################

# no planet
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
println("ln(evidence ratio): $(E2-E1)")

errs_glom, errs_kep, errs_off = GLOM_RV.GLOM_and_kep_and_offset_errs_from_hessian(H2, full_ks, current_offset)

println("Found planet vs injected one")
println(GLOM_RV.kep_and_offset_parms_str(full_ks, errs_kep, current_offset, errs_off))
println(GLOM_RV.kep_and_offset_parms_str(inject_ks, offset))

## Convert M0 to a T0 with GLOM_RV.M0_to_T0()
# GLOM_RV.M0_to_T0(full_ks.M0, errs_kep[3], u"d", full_ks.P; shift=0, ref_t = ref_t*u"d")