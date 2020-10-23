using Pkg
Pkg.activate(".")
Pkg.instantiate()

# for std()
using Statistics

# for importing the data from CSV
using DataFrames
using CSV

# For GLOM
import GPLinearODEMaker
GLOM = GPLinearODEMaker

# For this module
include("src/GLOM_RV_Example.jl")
GLOM_RV = Main.GLOM_RV_Example

#####################################################################
# CTRL+F "CHANGE"TO FIND PLACES WHERE YOU SHOULD MAKE MODIFICATIONS #
#####################################################################

# CHANGE: choose a kernel, I suggest 3 for Matern 5/2 or 4 for Quasi-periodic
# kernel
kernel_choice = 3
kernel_names = ["pp", "se", "m52", "qp", "m52_m52", "se_se"]
kernel_name = kernel_names[kernel_choice]
kernel_function, num_kernel_hyperparameters = include(pathof(GLOM) * "/../kernels/$(kernel_name)_kernel.jl")

# CHANGE: the stars rotation rate which is used as the first guess for some GLOM
# hyperparameters and starting point for priors
star_rot_rate = 17.  # days

# importing Yale's 101501 data
data = CSV.read("101501_activity.csv", DataFrame)

# CHANGE: observation times go here
obs_xs = collect(data[!, "Time [MJD]"])
# taking out the mean observation times makes the optimization easier for models
# with periodic parameters
GLOM_RV.remove_mean!(obs_xs)

# CHANGE: rvs and their errors go here
obs_rvs = data[!, "CCF RV [m/s]"]
obs_rvs_err = data[!, "CCF RV Error [m/s]"]

# CHANGE: activity indicators and thier errors go here
# you can actually have as many as you want, but obviously it will take longer
# to fit
obs_indicator1 = data[!, "CCF FWHM [m/s]"]
obs_indicator1_err = data[!, "CCF FWHM Err. [m/s]"]
obs_indicator2 = data[!, "BIS [m/s]"]
obs_indicator2_err = repeat([std(obs_indicator2)], length(obs_indicator2))  # I just put something here

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
# problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, ys; noise=noise, a0=[[1. 1 1];[1 1 1];[1 1 1]])
problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a0=[[1. 1 0];[1 0 1];[1 0 1]])

# Makes the std of each output equal to 1, improves fitting stability
# the normalizations are stored in problem_definition.normals
GLOM.normalize_problem_definition!(problem_definition)

# CHANGE: Setting initial fit values
initial_total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
initial_hypers = [[star_rot_rate], [star_rot_rate], [star_rot_rate], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1], [star_rot_rate, 2 * star_rot_rate, 1]]
append!(initial_total_hyperparameters, initial_hypers[kernel_choice])

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


fit_total_hyperparameters, result = GLOM_RV.fit_GLOM!(problem_definition, initial_total_hyperparameters, kernel_hyper_priors, add_kick!)
# fit_GLOM returns a vector of num_kernel_hyperparameters gp hyperparameters
# followed by the GLOM coefficients and the Optim result object

plot_xs = collect(LinRange(obs_xs[1]-10, obs_xs[end]+10, 300))
post, post_err, post_obs, post_obs_err = GLOM_RV.GLOM_posteriors(problem_definition, plot_xs, fit_total_hyperparameters)
GLOM_rvs_at_plot_xs, GLOM_ind1_at_plot_xs, GLOM_ind2_at_plot_xs = post
GLOM_rvs_err_at_plot_xs, GLOM_ind1_err_at_plot_xs, GLOM_ind2_err_at_plot_xs = post_err
GLOM_rvs_at_obs_xs, GLOM_ind1_at_obs_xs, GLOM_ind2_at_obs_xs = post_obs
GLOM_rvs_err_at_obs_xs, GLOM_ind1_err_at_obs_xs, GLOM_ind2_err_at_obs_xs = post_obs_err

activity_rvs = GLOM_rvs_at_obs_xs  # the best guess for activity RVs
clean_rvs = obs_rvs - activity_rvs  # the best guess for RVs without activity

# using Plots
# plt = scatter(obs_xs, obs_rvs, yerror=obs_rvs_err)
# plot!(plt, plot_xs, GLOM_rvs_at_plot_xs, ribbons=GLOM_rvs_err_at_plot_xs, fillalpha=0.3)
#
# plt = scatter(obs_xs, obs_indicator1, yerror=obs_indicator1_err)
# plot!(plt, plot_xs, GLOM_ind1_at_plot_xs, ribbons=GLOM_ind1_err_at_plot_xs, fillalpha=0.3)
#
# plt = scatter(obs_xs, obs_indicator2, yerror=obs_indicator2_err)
# plot!(plt, plot_xs, GLOM_ind2_at_plot_xs, ribbons=GLOM_ind2_err_at_plot_xs, fillalpha=0.3)


# # Could use this to iterate through all of the possible combinations of GLOM
# # coefficients
# possible_a0s = Matrix[]
# GLOM_RV.valid_a0s!(possible_a0s, zeros(n_out, n_dif))
# reverse!(possible_a0s)  # the expensive, filled out possibilities go first
# nℓs = Float64[]
# all_fit_total_hyperparameters = Vector[]
# problem_definitions = GLOM.GLO[]
# t0 = Libc.time()
# for i in 1:length(possible_a0s)
#     a0 = possible_a0s[i]
#     problem_definition = GLOM.GLO(kernel_function, num_kernel_hyperparameters, n_dif, n_out, obs_xs, copy(obs_ys); noise=copy(obs_noise), a0=a0)
#     GLOM.normalize_problem_definition!(problem_definition)
#     initial_total_hyperparameters = collect(Iterators.flatten(problem_definition.a0))
#     append!(initial_total_hyperparameters, initial_hypers[kernel_choice])
#     fit_total_hyperparameters, result = GLOM_RV.fit_GLOM!(problem_definition, initial_total_hyperparameters, kernel_hyper_priors, add_kick!; print_stuff=false)
#     append!(problem_definitions, [problem_definition])
#     append!(nℓs, result.minimum)
#     append!(all_fit_total_hyperparameters, [fit_total_hyperparameters])
#     println("\nDone with $(round(100 * i / length(possible_a0s); digits=2))% of a0 possibilities")
#     t = Libc.time() - t0
#     println("t: $(Int(round(t)))s t_left?: $(round(t * (length(possible_a0s) - i) / i / 60; digits=1)) mins")
# end

# best_fits = sortperm(nℓs)
#
# plot_xs = collect(LinRange(obs_xs[1]-10, obs_xs[end]+10, 300))
# post, post_err, post_obs, post_obs_err = GLOM_RV.GLOM_posteriors(problem_definitions[best_fits[1]], plot_xs, all_fit_total_hyperparameters[best_fits[1]])
# GLOM_rvs_at_plot_xs, GLOM_ind1_at_plot_xs, GLOM_ind2_at_plot_xs = post
# GLOM_rvs_err_at_plot_xs, GLOM_ind1_err_at_plot_xs, GLOM_ind2_err_at_plot_xs = post_err
# GLOM_rvs_at_obs_xs, GLOM_ind1_at_obs_xs, GLOM_ind2_at_obs_xs = post_obs
# GLOM_rvs_err_at_obs_xs, GLOM_ind1_err_at_obs_xs, GLOM_ind2_err_at_obs_xs = post_obs_err
#
# using Plots
# plt = scatter(obs_xs, obs_rvs, yerror=obs_rvs_err)
# plot!(plt, plot_xs, GLOM_rvs_at_plot_xs, ribbons=GLOM_rvs_err_at_plot_xs, fillalpha=0.3)
#
# plt = scatter(obs_xs, obs_indicator1, yerror=obs_indicator1_err)
# plot!(plt, plot_xs, GLOM_ind1_at_plot_xs, ribbons=GLOM_ind1_err_at_plot_xs, fillalpha=0.3)
#
# plt = scatter(obs_xs, obs_indicator2, yerror=obs_indicator2_err)
# plot!(plt, plot_xs, GLOM_ind2_at_plot_xs, ribbons=GLOM_ind2_err_at_plot_xs, fillalpha=0.3)
