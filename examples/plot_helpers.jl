function GLOM_plot(fig_loc, plot_xs, obs_xs, feat, err, GLOM_feat, GLOM_err, label, label_GLOM)
    plt = scatter(obs_xs, feat; yerror=err, label=label)
    plot!(plt, plot_xs, GLOM_feat; ribbons=GLOM_err, fillalpha=0.3, label=label_GLOM)
    png(plt, fig_loc)
end
function GLOM_ind_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs)
    for i in 2:length(rvs_and_inds)
        GLOM_plot(fig_dir * name_prefix * "ind$(i-1)", plot_xs, obs_xs, rvs_and_inds[i], rvs_and_inds_err[i], GLOM_at_plot_xs[i], GLOM_err_at_plot_xs[i], "obs ind $(i-1)", "GLOM ind $(i-1)")
    end
end
function GLOM_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs)
    GLOM_plot(fig_dir * name_prefix * "rv", plot_xs, obs_xs, rvs_and_inds[1], rvs_and_inds_err[1], GLOM_at_plot_xs[1], GLOM_err_at_plot_xs[1], "obs RVs", "GLOM RVs")
    GLOM_ind_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs)
end
function GLOM_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs, rvs_np)
    GLOM_plot(fig_dir * name_prefix * "rv_nokep", plot_xs, obs_xs, rvs_np, rvs_and_inds_err[1], GLOM_at_plot_xs[1], GLOM_err_at_plot_xs[1], "obs-Kep RVs", "GLOM RVs")
    GLOM_plot(fig_dir * name_prefix * "rv", plot_xs, obs_xs, rvs_and_inds[1], rvs_and_inds_err[1], GLOM_at_plot_xs[1] + ustrip.(current_ks.(plot_xs.*u"d")), GLOM_err_at_plot_xs[1], "obs RVs", "GLOM+Kep RVs")
    GLOM_ind_plots(name_prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs)
end
function plot_helper(prefix::String, ks::GLOM_RV.KeplerSignal, fit_total_hyperparameters)
    GLOM_at_plot_xs, GLOM_err_at_plot_xs, GLOM_at_obs_xs = GLOM_RV.GLOM_posteriors(glo, plot_xs, fit_total_hyperparameters; y_obs = GLOM_RV.remove_kepler(glo_rv, current_ks))
    rvs_np = GLOM_RV.remove_kepler(GLOM_RV.get_rv(glo_rv), glo_rv.time, current_ks)
    GLOM_plots(prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs, rvs_np)

    plot_kep_xs = collect(LinRange(0, ustrip(best_period), 1000))
    scatter(remainder(glo.x_obs, ustrip(best_period)), ustrip.(GLOM_RV.get_rv(glo_rv)) - GLOM_at_obs_xs[1]; yerror=ustrip.(GLOM_RV.get_rv_noise(glo_rv)), label="data")
    plot!(plot_kep_xs, ustrip.(current_ks.(plot_kep_xs.*u"d")); label="kep")
    png(fig_dir * prefix * "rv_phase")
end
function plot_helper(prefix::String, fit_total_hyperparameters)
    GLOM_at_plot_xs, GLOM_err_at_plot_xs, GLOM_at_obs_xs = GLOM_RV.GLOM_posteriors(glo, plot_xs, fit_total_hyperparameters)
    GLOM_plots(prefix, plot_xs, obs_xs, rvs_and_inds, rvs_and_inds_err, GLOM_at_plot_xs, GLOM_err_at_plot_xs)
    _, _, GLOM_at_obs_xs_inf = GLOM_RV.GLOM_posteriors(glo, [1., 2.], fit_total_hyperparameters; inflate_errors=1)
    scatter(obs_rvs, GLOM_at_obs_xs[1];
        title = star_str * ", obs RV std: $(round(std(obs_rvs); digits=3))",
        xlabel = "Observed RVs (m/s)",
        ylabel = "GLOM RVs (m/s)",
        label =  "GLOM posterior, resid std        : $(round(std(obs_rvs - GLOM_at_obs_xs[1]); digits=3)), ρ: $(round(cor(obs_rvs, GLOM_at_obs_xs[1]); digits=3))",
        legend=:topleft)
    scatter!(obs_rvs, GLOM_at_obs_xs_inf[1];
        label = "\" with inflated errors, resid std: $(round(std(obs_rvs - GLOM_at_obs_xs_inf[1]); digits=3)), ρ: $(round(cor(obs_rvs, GLOM_at_obs_xs_inf[1]); digits=3))")
    png(fig_dir * prefix * "inflated_errs")
end
