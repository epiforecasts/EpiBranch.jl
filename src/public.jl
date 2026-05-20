# Julia 1.11+ public declarations — extensible API that is not exported.
# These functions form the intervention interface: package users subtyping
# AbstractIntervention should override them, but they are not brought into
# scope by `using EpiBranch`.

public initialise_individual!
public resolve_individual!
public apply_post_transmission!
public competing_risk
public cap_offspring
public reset!
public required_fields
