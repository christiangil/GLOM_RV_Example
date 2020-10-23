const n_kep_parms = 6

include("general_functions.jl")
include("problem_definition_functions.jl")
include("RV_functions.jl")
include("keplerian_derivatives.jl")
include("prior_functions.jl")
include("diagnostic_functions.jl")
include("plotting_functions.jl")

const light_speed = uconvert(u"m/s",1u"c")
const light_speed_nu = ustrip(light_speed)
