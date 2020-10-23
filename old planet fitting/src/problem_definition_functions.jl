# problem_definition_functions.jl
using Dates
using Unitful
using UnitfulAstro
import GPLinearODEMaker

"""
A structure that holds all of the relevant information for constructing the
model used in the Jones et al. 2017+ paper (https://arxiv.org/pdf/1711.01318.pdf).
"""
struct GLO_RV{T1<:Real, T2<:Integer}
	GLO::GPLinearODEMaker.GLO  # kernel function
	time::Vector{T} where T<:Unitful.Time
	time_unit::Unitful.FreeUnits  # the units of x_obs
	rv::Vector{T} where T<:Unitful.Velocity
	rv_unit::Unitful.FreeUnits  # the units of the RV section of y_obs
	rv_noise::Vector{T} where T<:Unitful.Velocity # the measurement noise at all observations

	function GLO_RV(
		prob_def::GPLinearODEMaker.GLO;
		time_unit = u"d",
		rv_unit = u"m/s")
		return new{typeof(prob_def.x_obs[1]),typeof(prob_def.n_kern_hyper)}(prob_def, prob_def.x_obs .* time_unit, time_unit, prob_def.y_obs[1:prob_def.n_out:end] .* rv_unit, rv_unit, prob_def.noise[1:prob_def.n_out:end] .* rv_unit)
	end
end
