## This script shows how one could use GLOM to find a planet looking for an activity lag, δ, and a stellar jitter term

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

plot_dir = "examples/figs/lag_and_jitter/"

####################
# Simulation setup #
####################

# what GLOM coefficients are being used (for the lag version of GLOM, don't suggest using derivatives)
a0 = reshape([1., 1], (2,1))
# a0 = reshape([0., 1, 1, 0], (2,2))
n_coeff = length(GLOM.remove_zeros(collect(Iterators.flatten(a0))))
# Observation times
obs_xs = sort!(append!(LinRange(1,30., 30) .+ 0.1.*rand(30), LinRange(1,30., 30) .+ 0.1.*rand(30)))  # twice nightly observations for a 30 days
# Lengthscale of the input GLOM model
λ = 10.  # days
δ = -3.  # days
jitter = 1.  # m/s
truth_hypers = [λ, δ, jitter]
initial_hypers = copy(truth_hypers)
# Input RV noise
rv_noise = 0.1 .* ones(length(obs_xs))  # 10 cm/s RV noise
# Input indicator noise
indicator_noise = 0.3 .* ones(length(obs_xs))
# Input planet
inject_ks = GLOM_RV.kep_signal(; K=1.0u"m/s", P=sqrt(2)*5u"d", M0=3.2992691080593275, ω_or_k=4.110936513051912, e_or_h=0.1)

# the observations have no offset
n_obs = length(obs_xs)
offset = GLOM_RV.basic_Offset(n_obs)
current_offset = copy(offset)

#################
# Custom kernel #
#################

# Creating a custom kernel that models two outputs as a shared latent GP with a time separation with an additional white noise (jitter) term for the first output
kernel_name = "m52"
kernel_mat, n_kern_hyper_mat = GLOM.include_lag_kernel(kernel_name)
kernel_scale, n_kern_hyper_scale = GLOM.include_kernel("scale")

add_rv_jitter = true
add_activity_jitter = false && add_rv_jitter  # currently built assuming that if you are using activity jitter, you are also using RV jitter, activity jitter not used in this example
n_non_jitter = 2 + n_kern_hyper_mat  # the 2 is related to whether you are taking a derivative w.r.t. t or t', not the number of coefficients
n_jitter = add_rv_jitter + add_activity_jitter
n_hyper_and_time = n_non_jitter + n_jitter
rv_jitter_d_ind = n_non_jitter + add_rv_jitter
act_jitter_d_ind = rv_jitter_d_ind + add_activity_jitter
rv_jitter_ind = rv_jitter_d_ind - 2
act_jitter_ind = act_jitter_d_ind - 2
jitter_d_inds = rv_jitter_d_ind:act_jitter_d_ind

time_and_jitter_d_inds_rv = append!(collect(1:2), [rv_jitter_d_ind])  # indices for the t, t' and jitter hyperparameters derivatives
other_d_inds_rv = [i for i in 1:n_hyper_and_time if !(i in time_and_jitter_d_inds_rv)]
# time_and_jitter_d_inds_act = append!(collect(1:2), [act_jitter_d_ind])  # indices for the coefficient and the jitter hyperparameters
# other_d_inds_act = [i for i in 1:n_hyper_and_time if !(i in time_and_jitter_d_inds_act)]

function kernel_function(hyperparameters, δ, dorder; outputs::AbstractVector{<:Integer}=[1,1], kwargs...)
    res = 0
    # if there are no jitter term derivatives being taken
    if all(view(dorder, jitter_d_inds) .< 1)
        res += kernel_mat(view(hyperparameters, 1:n_kern_hyper_mat), δ, view(dorder, 1:(2+n_kern_hyper_mat)); outputs=outputs, kwargs...)
    end
    # if you are on the output with RV jitter and no non-jitter derivatives are being taken
    if all(outputs.==1) && all(view(dorder, other_d_inds_rv) .< 1)
        res += Int(δ == 0)*kernel_scale(view(hyperparameters, rv_jitter_ind:rv_jitter_ind), δ, view(dorder, time_and_jitter_d_inds_rv))
    end
    # # if you are on the output with activity jitter and no non-jitter derivatives are being taken
    # if all(outputs.==2) && all(view(dorder, other_d_inds_act) .< 1)
    #     res += Int(δ == 0)*kernel_scale(view(hyperparameters, act_jitter_ind:act_jitter_ind), δ, view(dorder, time_and_jitter_d_inds_act))
    # end
    return res
end
num_kernel_hyperparameters = n_kern_hyper_mat + n_jitter

# defining functions for GP coefficient and hyperparameter priors and how to kick the fitting out of saddle points (see other kernel_hyper_priors and add_kick functions in fit_GLOM_functions.jl)
tighten_lengthscale_priors = 1
function hyper_npriors(total_hyper::Vector, d::Integer)
    npriors = append!(zeros(n_coeff), -[GLOM.log_gamma(total_hyper[n_coeff+1], GLOM.gamma_mode_std_to_α_θ(initial_hypers[1], initial_hypers[1]/tighten_lengthscale_priors); d=d), GLOM.log_uniform(total_hyper[n_coeff+2]; min_max=[-1000,1000], d=d), GLOM.log_loguniform(total_hyper[n_coeff+3], [1e-3,10]; d=d)])
    @assert length(npriors) == length(total_hyper)
    if d==0; return sum(npriors) end
    if d==1; return npriors end
    if d==2; return Diagonal(npriors) end
    @error d_err_msg
end
function add_kick!(kernel_hyper::AbstractVector{<:Real})
    @assert length(kernel_hyper) == length(truth_hypers)
    kernel_hyper[1] *= GLOM_RV.centered_rand(; center=1, scale=0.4)
    kernel_hyper[2] *= GLOM_RV.centered_rand(; center=1, scale=0.2)
    kernel_hyper[3] *= GLOM_RV.centered_rand(; center=1, scale=0.5)
    return kernel_hyper
end

###################
# Data simulation #
###################

# simulating activity and RVs from the GLOM prior
initial_total_hyperparameters = collect(Iterators.flatten(a0))
append!(initial_total_hyperparameters, initial_hypers)
Σ = GLOM.prior_covariance(kernel_function, initial_hypers, a0, obs_xs; kernel_changes_with_output=true)
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

glo = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a=copy(a0), kernel_changes_with_output=true)
GLOM.normalize_GLO!(glo)

##################################
# Searching for a good lag guess #
##################################

# creating a workspace that knows there will be some assymetries from the lag parameter
workspace = GLOM.nlogL_matrix_workspace(glo, append!(collect(Iterators.flatten(glo.a)), initial_hypers); ignore_asymmetry=true)

# find a first guess for lag via grid search
potential_lags = LinRange(-10,20,200)
potential_lags_ℓ = [GLOM.nlogL_GLOM!(workspace, glo, append!(collect(Iterators.flatten(glo.a)), [initial_hypers[1], i, initial_hypers[3]])) for i in potential_lags]
plt = GLOM_RV.periodogram_plot(collect(potential_lags), potential_lags_ℓ; font_size=14, title=L"\textrm{Lag \ periodogram}", xlabel=L"\delta", truth=δ)
png(plot_dir * "lag_search")

potential_lags_post = potential_lags[GLOM_RV.find_modes(-potential_lags_ℓ)]
# initial_hypers[2] = potential_lags_post[findfirst(potential_lags_post .> 0.5)]
initial_hypers[2] = potential_lags_post[1]

total_hyperparameters = append!(collect(Iterators.flatten(glo.a)), initial_hypers)

##################
# Fit GLOM model #
##################

# fitting the GLOM model to the simulated data
fit1_total_hyperparameters, result = GLOM_RV.fit_GLOM!(workspace, glo, initial_total_hyperparameters, hyper_npriors, add_kick!)

# just a helper function to make it easier to evaluate the priors on the hyperparameters
fit_nlogL1 = GLOM.nlogL_GLOM(glo, fit1_total_hyperparameters)
H1 = (GLOM.∇∇nlogL_GLOM(glo, fit1_total_hyperparameters)
    + hyper_npriors(GLOM.remove_zeros(fit1_total_hyperparameters), 2))
fit1_total_hyperparameters_σ = GLOM_RV.errs_from_hessian(H1)

title_helper(nzhypers, hypers_σ, ℓ) = L" \lambda_{M^5/_2}=%$(GLOM.rounded(nzhypers[3])) \pm %$(GLOM.rounded(hypers_σ[3])) \ \textrm{days}, \ \delta=%$(GLOM.rounded(nzhypers[4])) \pm %$(GLOM.rounded(hypers_σ[4])) \ \textrm{days}, \ \sigma_\star=%$(GLOM.rounded(nzhypers[5] * glo.normals[1] * nzhypers[1])) \pm %$(GLOM.rounded(hypers_σ[5] * glo.normals[1] * nzhypers[1])) \ \textrm{m/s}, \ \ell=%$(GLOM.rounded(ℓ))"
x_samp = GLOM_RV.plot_points(glo; max_extra=100, ignore_obs=true)
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
initial_uE = -initial_nlogL - hyper_npriors(initial_total_hyperparameters, 0)
println(initial_uE, "\n")

println("ending hyperparameters")
println(fit1_total_hyperparameters)
uE1 = -fit_nlogL1 - hyper_npriors(fit1_total_hyperparameters, 0)
println(uE1, "\n")

Σ_obs = GLOM.Σ_observations(glo, fit1_total_hyperparameters)

#######################################
# Keplerian periodogram planet search #
#######################################

# search for planet period with a keplerian peridogram
period_grid, likelihoods, unnorm_posteriors, kss, offsets = GLOM_RV.keplerian_periodogram(glo_rv, current_offset, fit1_total_hyperparameters; Σ_obs=Σ_obs, nlogprior_kernel=hyper_npriors(GLOM.remove_zeros(fit1_total_hyperparameters), 0))
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
    fit1_total_hyperparameters, hyper_npriors, add_kick!, current_ks, current_offset;
    avoid_saddle=false, fit_alpha=1e-3)

# convert the semi-linear keplerian model to a fully non-linear one
full_ks = GLOM_RV.kep_signal(current_ks)

fit_nlogL2 = GLOM.nlogL_GLOM(glo, fit2_total_hyperparameters; y_obs=GLOM_RV.remove_kepler(glo_rv, full_ks, current_offset))
uE2 = -fit_nlogL2 - hyper_npriors(fit2_total_hyperparameters, 0) + GLOM_RV.logprior_kepler(full_ks; use_hk=true)

H2 = Matrix(GLOM_RV.∇∇nlogL_GLOM_and_kep(glo_rv, fit2_total_hyperparameters, full_ks, current_offset; include_kepler_priors=true))
n_hyper = length(GLOM.remove_zeros(fit2_total_hyperparameters))
H2[1:n_hyper, 1:n_hyper] += hyper_npriors(GLOM.remove_zeros(fit2_total_hyperparameters), 2)
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