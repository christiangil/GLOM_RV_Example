using Unitful
using Statistics
function GLOM_plot(fig_loc::String, plot_xs::AbstractVector, obs_xs::AbstractVector, feat::AbstractVector, err::AbstractVector, GLOM_feat::AbstractVector, GLOM_err::AbstractVector, ylabel::String; label_feat::String="obs", label_GLOM::String="GLOM", kwargs...)
    plt = scatter(obs_xs, feat; yerror=err, label=label_feat,
        xlabel = "Time (days)",
        ylabel = ylabel,
        kwargs...)
    plot!(plt, plot_xs, GLOM_feat; ribbons=GLOM_err, fillalpha=0.3, label=label_GLOM)
    png(plt, fig_loc)
end
function GLOM_ind_plots(name_prefix::String, plot_xs::AbstractVector, obs_xs::AbstractVector, rvs_and_inds::AbstractVector{<:AbstractVector}, rvs_and_inds_err::AbstractVector{<:AbstractVector}, GLOM_at_plot_xs::AbstractVector{<:Vector}, GLOM_err_at_plot_xs::AbstractVector{<:AbstractVector}; kwargs...)
    for i in 2:length(rvs_and_inds)
        GLOM_plot(fig_dir * name_prefix * "ind$(i-1)", plot_xs, obs_xs, rvs_and_inds[i], rvs_and_inds_err[i], GLOM_at_plot_xs[i], GLOM_err_at_plot_xs[i], "ind $(i-1)"; kwargs...)
    end
end
function GLOM_plots(name_prefix::String, plot_xs::AbstractVector, obs_xs::AbstractVector, rvs_and_inds::AbstractVector{<:AbstractVector}, rvs_and_inds_err::AbstractVector{<:AbstractVector}, GLOM_at_plot_xs::AbstractVector{<:Vector}, GLOM_err_at_plot_xs::AbstractVector{<:AbstractVector}; kwargs...)
    GLOM_plot(fig_dir * name_prefix * "rv", plot_xs, obs_xs, rvs_and_inds[1], rvs_and_inds_err[1], GLOM_at_plot_xs[1], GLOM_err_at_plot_xs[1], "RV (m/s)"; kwargs...)
    GLOM_ind_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs; kwargs...)
end
function GLOM_plots(name_prefix::String, plot_xs::AbstractVector, obs_xs::AbstractVector, rvs_and_inds::AbstractVector{<:AbstractVector}, rvs_and_inds_err::AbstractVector{<:AbstractVector}, GLOM_at_plot_xs::AbstractVector{<:Vector}, GLOM_err_at_plot_xs::AbstractVector{<:AbstractVector}, ks::GLOM_RV.KeplerSignal; kwargs...)
    rvs_np = GLOM_RV.remove_kepler(GLOM_RV.get_rv(glo_rv), glo_rv.time, ks)
    GLOM_plot(fig_dir * name_prefix * "rv_nokep", plot_xs, obs_xs, rvs_np, rvs_and_inds_err[1], GLOM_at_plot_xs[1], GLOM_err_at_plot_xs[1], "RV (m/s)"; label_feat="obs-Kep", kwargs...)
    GLOM_plot(fig_dir * name_prefix * "rv", plot_xs, obs_xs, rvs_and_inds[1], rvs_and_inds_err[1], GLOM_at_plot_xs[1] + ustrip.(ks.(plot_xs.*u"d")), GLOM_err_at_plot_xs[1], "RV (m/s)"; label_GLOM="GLOM+Kep", kwargs...)
    GLOM_ind_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs; kwargs...)
end
function plot_helper(glo_rv::GLOM_RV.GLO_RV, plot_xs::AbstractVector, obs_xs::AbstractVector, rvs_and_inds::AbstractVector{<:AbstractVector}, rvs_and_inds_err::AbstractVector{<:AbstractVector}, prefix::String, ks::GLOM_RV.KeplerSignal, fit_total_hyperparameters::AbstractVector; kwargs...)
    GLOM_at_plot_xs, GLOM_err_at_plot_xs, GLOM_at_obs_xs = GLOM_RV.GLOM_posteriors(glo_rv.GLO, plot_xs, fit_total_hyperparameters; y_obs = GLOM_RV.remove_kepler(glo_rv, ks))
    GLOM_plots(prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs, ks; kwargs...)

    plot_kep_xs = collect(LinRange(0, ustrip(best_period), 1000))
    scatter(remainder(glo.x_obs, ustrip(best_period)), ustrip.(GLOM_RV.get_rv(glo_rv)) - GLOM_at_obs_xs[1]; yerror=ustrip.(GLOM_RV.get_rv_noise(glo_rv)), label="data", kwargs...)
    plot!(plot_kep_xs, ustrip.(ks.(plot_kep_xs.*u"d")); label="kep")
    png(fig_dir * prefix * "rv_phase")
end
function plot_helper(glo::GLOM.GLO, plot_xs::AbstractVector, obs_xs::AbstractVector, obs_rvs::AbstractVector, rvs_and_inds::AbstractVector{<:AbstractVector}, rvs_and_inds_err::AbstractVector{<:AbstractVector}, prefix::String, fit_total_hyperparameters::AbstractVector; kwargs...)
    GLOM_at_plot_xs, GLOM_err_at_plot_xs, GLOM_at_obs_xs = GLOM_RV.GLOM_posteriors(glo, plot_xs, fit_total_hyperparameters)
    GLOM_plots(prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs; kwargs...)
    GLOM_at_obs_xs_inf = GLOM_RV.GLOM_posteriors(glo, fit_total_hyperparameters; inflate_errors=1)
    scatter(obs_rvs, GLOM_at_obs_xs[1];
        xlabel = "Observed RVs (m/s)",
        ylabel = "GLOM RVs (m/s)",
        label =  "GLOM posterior, resid std        : $(round(std(obs_rvs - GLOM_at_obs_xs[1]); digits=3)), ρ: $(round(cor(obs_rvs, GLOM_at_obs_xs[1]); digits=3))",
        legend=:topleft,
        kwargs...)
    scatter!(obs_rvs, GLOM_at_obs_xs_inf[1];
        label = "\" with inflated errors, resid std: $(round(std(obs_rvs - GLOM_at_obs_xs_inf[1]); digits=3)), ρ: $(round(cor(obs_rvs, GLOM_at_obs_xs_inf[1]); digits=3))")
    png(fig_dir * prefix * "inflated_errs")
end
