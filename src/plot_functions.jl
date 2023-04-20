using Plots
using Unitful
using Statistics
using LaTeXStrings


function plot_points(glo::GLOM.GLO; max_extra::Int=500, ignore_obs::Bool=false)
    n_samp_points = convert(Int64, max(max_extra, round(2 * sqrt(2) * length(glo.x_obs))))
    x_samp = sort(append!(collect(range(minimum(glo.x_obs)-5; stop=maximum(glo.x_obs)+5, length=n_samp_points)), [(glo.x_obs[i] + glo.x_obs[i - 1]) / 2 for i in 2:length(glo.x_obs)]))
    if !ignore_obs; append!(x_samp, glo.x_obs) end
    sort!(x_samp)
    return x_samp
end


function make_subplot(glo::GLOM.GLO, x_samp::Vector, mean_GP::Vector, mean_GP_σ::Vector, output::Integer, show_curves::Matrix; label="", x_mean::Real=0, y_obs=glo.y_obs, kwargs...)
    sample_output_indices = output:glo.n_out:length(x_samp) * glo.n_out
    obs_output_indices = output:glo.n_out:length(y_obs)
    f = glo.normals[output]
    size(show_curves, 1) > 0 ? mean_alpha = 0. : mean_alpha = 1.
    plt = scatter(glo.x_obs .+ x_mean, f*y_obs[obs_output_indices], yerror=f*glo.noise[obs_output_indices]; msw=0.5, label=label, kwargs...)
    plot!(plt, x_samp .+ x_mean, f*mean_GP[sample_output_indices]; ribbon=f*mean_GP_σ[sample_output_indices], alpha=mean_alpha, label="")
    for i in axes(show_curves, 1)
        plot!(plt, x_samp .+ x_mean, f*show_curves[i, sample_output_indices], label="", alpha=0.6)
    end
    return plt
end


function make_plot(glo::GLOM.GLO, total_hyperparameters::Vector, subtitles::Vector{<:AbstractString}, ylabels::Vector{<:AbstractString}; x_samp::Vector=plot_points(glo), title="", font_size=10, thickness_scaling=2, n_show::Int=0, kwargs...)
    mean_GP, mean_GP_σ, mean_GP_obs, Σ = GLOM.GP_posteriors(glo, x_samp, total_hyperparameters; return_mean_obs=true)
    n_total_samp_points = length(x_samp) * glo.n_out
    show_curves = zeros(n_show, n_total_samp_points)
    L = GLOM.ridge_chol(Σ).L
    for i in 1:n_show
        show_curves[i, :] = L * randn(n_total_samp_points) + mean_GP
    end
    plots = [plot(title = title, grid = false, showaxis = false, ticks=false, bottom_margin = -30Plots.px)]
    for i in 1:glo.n_out
        i == glo.n_out ? x_label = L"\textrm{Time}" : x_label=""
        append!(plots, [make_subplot(glo, x_samp, mean_GP, mean_GP_σ, i, show_curves; title=subtitles[i], ylabel=ylabels[i], xlabel=x_label, kwargs...)])
    end
    plot(plots...;
        layout=@layout([A{0.01h}; grid(2,1)]), size=(1800,900), titlefontsize=Int(round(1.25*font_size)), xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size, thickness_scaling=thickness_scaling)
end


function make_plot(glo_rv::GLO_RV, ks::KeplerSignal, offset::Offset, total_hyperparameters::Vector, subtitles::Vector{<:AbstractString}, ylabels::Vector{<:AbstractString}; x_samp::Vector=plot_points(glo), title="", font_size=10, thickness_scaling=2, n_show::Int=0, kwargs...)
    glo = glo_rv.GLO
    y_obs = remove_kepler(glo_rv, ks, offset)
    mean_GP, mean_GP_σ, mean_GP_obs, Σ = GLOM.GP_posteriors(glo, x_samp, total_hyperparameters; return_mean_obs=true, y_obs=y_obs)
    n_total_samp_points = length(x_samp) * glo.n_out
    show_curves = zeros(n_show, n_total_samp_points)
    L = GLOM.ridge_chol(Σ).L
    for i in 1:n_show
        show_curves[i, :] = L * randn(n_total_samp_points) + mean_GP
    end
    plots = [plot(title = title, grid = false, showaxis = false, ticks=false, bottom_margin = -30Plots.px)]
    for i in 1:glo.n_out
        i == glo.n_out ? x_label = L"\textrm{Time}" : x_label=""
        append!(plots, [make_subplot(glo, x_samp, mean_GP, mean_GP_σ, i, show_curves; title=subtitles[i], ylabel=ylabels[i], xlabel=x_label, y_obs=y_obs, kwargs...)])
    end
    plot(plots...;
        layout=@layout([A{0.01h}; grid(2,1)]), size=(1800,900), titlefontsize=Int(round(1.25*font_size)), xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size, thickness_scaling=thickness_scaling)
end


nicer_plot(; font_size=10, thickness_scaling=2, kwargs...) = plot(; size=(1800,900), titlefontsize=Int(round(1.25*font_size)), xtickfontsize=font_size, ytickfontsize=font_size, xguidefontsize=font_size, yguidefontsize=font_size, legendfontsize=font_size, thickness_scaling=thickness_scaling, kwargs...)


function periodogram_plot(period_grid::Vector, likelihoods::Vector; ylabel=L"\ell", title=L"\textrm{Keplerian \ Periodogram}", xlabel=L"\textrm{Period}", truth=nothing, kwargs...)
    plt = nicer_plot(; title=title, kwargs...)
    plot!(plt, ustrip.(period_grid), likelihoods; label="", ylabel=ylabel, xlabel=xlabel)
    if !isnothing(truth)
        vline!([truth]; label="")
    end
    return plt
end


function keplerian_plot(glo_rv::GLO_RV, total_hyperparameters, ks::KeplerSignal, offset::Offset; label="", ylabel=L"\textrm{RV \ (m/s)}", x_mean::Real=0, x_samp::Vector=plot_points(glo_rv.GLO), kwargs...)
    glo = glo_rv.GLO
    y_obs = glo.y_obs
    mean_GP_obs, mean_GP_obs_σ, Σ = GLOM.GP_posteriors(glo, glo.x_obs, total_hyperparameters; y_obs=remove_kepler(glo_rv, ks, offset))
    obs_output_indices = 1:glo.n_out:length(y_obs)
    f = glo.normals[1]
    plt = nicer_plot(; title=L"\textrm{Keplerian \ Model}", kwargs...)
    scatter!(plt, glo.x_obs .+ x_mean, f*(y_obs[obs_output_indices] .- mean_GP_obs[obs_output_indices]) - ustrip.(offset()), yerror=f*glo.noise[obs_output_indices]; msw=0.5, label=L"\textrm{Data - Offset - GLOM}", ylabel=ylabel, kwargs...)
    plot!(plt, x_samp .+ x_mean, ustrip.(ks.(unit(glo_rv.time[1]) .* x_samp)); label="", xlabel=L"\textrm{Time}")

    plot_kep_xs = collect(LinRange(0, ustrip(ks.P), 1000))
    plt_phase = nicer_plot(; title=L"\textrm{Keplerian \ Model}", kwargs...)
    scatter!(plt_phase, GLOM.remainder(convert_and_strip_units.(unit(ks.P), glo_rv.time), ustrip(ks.P)), f*(y_obs[obs_output_indices] .- mean_GP_obs[obs_output_indices]) - ustrip.(offset()), yerror=f*glo.noise[obs_output_indices]; msw=0.5, label=L"\textrm{Data - Offset - GLOM}", ylabel=ylabel, kwargs...)
    plot!(plt_phase, plot_kep_xs, convert_and_strip_units.(unit(glo_rv.rv_factor), ks.(plot_kep_xs .* unit(ks.P))); label="", xlabel=L"\textrm{Time \ (phased)}")
    return plt, plt_phase
end
    
