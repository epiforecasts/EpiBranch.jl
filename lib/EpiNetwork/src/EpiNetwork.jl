module EpiNetwork

using EpiBranch
using Distributions
using Random

# `NetworkProcess` describes the transmission alone over EpiBranch's continuous-time
# simulation surface: it builds a population with `new_state`/`add_individuals!`
# and runs the shared `simulate_once` seam, deriving the infectious window from the
# progression composed onto it with a `ModelSpec`. `import` is needed for the
# methods we add to (`population_size`, `simulate_once`); the rest are pulled in
# because they are not brought into scope by `using EpiBranch`.
import EpiBranch: population_size, new_state, add_individuals!, apply_observation!,
    simulate_once, SimOpts, infectious_from,
    _honours_termination_controls, _validate_process_windows,
    _warn_uncovered_terminal_states, _state_real
# The infection layer is built and read out of a simulation by EpiBranch's
# helpers, which close each node's window where the simulator's race does. The
# simulator and the likelihood share EpiBranch's community-hazard helpers as
# well, and agree on when that term applies and what it is.
import EpiBranch: _infection_layer_columns, _infection_layer_fields,
    _layer_host_times, _ext_active, _ext_survival_of, _ext_prepare,
    _ext_draw, _ext_partype, _check_external, _normalise_external
# The routed likelihood leaves each route's removal to the same lapsing-stretch
# record the single-route likelihood reads, gated per route, and evaluates a
# susceptibility effect over every route and the external hazard the same way
# the single-route likelihood evaluates it over a contact structure's rows.
import EpiBranch: _removal_gaps, _host_stretches, _gapped_cumhazard, _removed_at,
    _validate_infection_likelihood, susceptibility_components, _scaled_cumhazard,
    _scaled_loghazard, _mixtures_partype, _layer_host, _is_minus_inf,
    INTERVENTION_REMOVAL, followup_end, contact_structure,
    cumhazard, loghazard, logsumexp

export NetworkProcess, RoutedNetwork
export NetworkInfections, network_infections
export RoutedNetworkInfections, routed_network_infections, RoutedMassActionHazard
export pairwise_surv_loglik, compile_contact_pairs

include("network_process.jl")
include("routed_network.jl")
include("network_simulate.jl")
include("likelihood.jl")
include("routed_likelihood.jl")

end # module
