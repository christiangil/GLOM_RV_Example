# Example for Jacob
using Pkg
Pkg.activate("examples")
Pkg.instantiate()
using Unitful; using UnitfulAstro
using LinearAlgebra
using GLOM_RV_Example; GRV = GLOM_RV_Example

print_stuff = true
data_unit = 1u"m/s"
times = sort(10u"d" .* rand(100))
data = sin.(ustrip.(times))  # m/s
σ = repeat([0.3], length(times))  # m/s
Σ = Diagonal(σ.^2)
period_guess = 6.3u"d"

fit_ks_epi = GRV.fit_kepler_epicyclic(data, times, Σ, period_guess; data_unit=data_unit, print_stuff=print_stuff)
fit_ks = GRV.fit_kepler(data, times, Σ, GRV.kep_signal_wright(0u"m/s", fit_ks_epi.P, fit_ks_epi.M0, minimum([fit_ks_epi.e, 0.3]), 0, fit_ks_epi.γ);
    data_unit=data_unit, print_stuff=print_stuff, avoid_saddle=true, include_priors=true)

println(GRV.kep_parms_str(fit_ks))
