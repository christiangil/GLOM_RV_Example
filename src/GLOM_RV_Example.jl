module GLOM_RV_Example

const n_kep_parms = 6
include("general_functions.jl")
include("fit_GLOM_functions.jl")
include("problem_definition_functions.jl")
export GLO_RV
include("RV_functions.jl")
include("keplerian_derivatives.jl")
include("prior_functions.jl")

end # module
