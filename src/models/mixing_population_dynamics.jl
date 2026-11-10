# ── Approximate dynamics for a MixingProcess ─────────────────────────
#
# The multi-group analogue of `MeanFieldProcess`/`LinearNoiseProcess`
# (population_dynamics.jl): susceptible, infectious and removed compartments
# per mixing group rather than one set for the whole population, the
# large-population limit of a `MixingProcess` composed with a single
# exponential removal step. Built directly from the `ModelSpec` it
# approximates, reading the group sizes off the population its `attributes`
# assign, the between-group rate and calendar off the `MixingProcess` itself,
# and a `GroupVaccination` campaign off its interventions
# (mixing_vaccination.jl), so the model is never written down twice. Shares
# `simulate(model, state; max_time)` with the scalar approximations and the
# exact models, so `HybridProcess` (hybrid.jl) can switch to either from a
# `MixingProcess` run.
#
# Each group keeps its susceptibles `S` who have had no dose, `I` infectious,
# `V` dosed susceptibles (whose hazard is `hazard` times that of `S`: below one
# for a leaky vaccine, one for the all-or-nothing vaccine's failures), `P`
# the immune (from before the outbreak, or from an all-or-nothing dose that
# took), and `D`, the doses given so far, whatever their recipient's status.
# What has been removed is what is left of the group's size, `R`.

# A member's mixing group, read off its state the same missing-safe way
# `_MassAction` reads it when the exact race is built (mass_action.jl), so a
# person the attributes left without one of `mixing_by`'s keys falls into a
# `missing` group here too, rather than erroring in one path and not the
# other.
_group_key(ind, mixing_by::Tuple) = Tuple(get(ind.state, k, missing) for k in mixing_by)

# The groups `spec`'s `attributes` assign and how many people land in each,
# realised by actually building the population (`new_state` plus
# `add_individuals!`, with nobody infected and no interventions), so a group
# assignment that depends on anything `attributes` can see — an individual's
# id, say — is read exactly as the exact model would read it. A stochastic
# assignment would make the group sizes themselves random on every draw,
# which a fixed-size compartment cannot represent, so `attributes` is
# expected to assign groups deterministically.
function _mixing_group_layout(spec::ModelSpec, mixing_by::Tuple, rng::AbstractRNG)
    bare = process(spec)
    state = new_state(bare, progression(spec), attributes(spec), rng)
    add_individuals!(state, population_size(bare), (); setup = (ind, i) -> nothing)
    keys = Tuple[]
    index = Dict{Any, Int}()
    sizes = Int[]
    for ind in state.individuals
        key = _group_key(ind, mixing_by)
        g = get!(index, key) do
            push!(keys, key)
            push!(sizes, 0)
            length(keys)
        end
        sizes[g] += 1
    end
    return keys, sizes
end

# `spec` must be a `ModelSpec` built on a `MixingProcess`, so there is both a
# `rate`/`calendar` to read and an `attributes` that assigns the groups
# `rate` is defined over. Its interventions are read by `_read_campaign`,
# which refuses any it cannot fold into the compartments rather than leave
# them out.
function _mixing_process_spec(spec, what)
    spec isa ModelSpec && process(spec) isa MixingProcess || throw(
        ArgumentError(
            "$what needs a ModelSpec built on a MixingProcess, with `attributes` " *
                "assigning the keys its `mixing_by` groups by, so the group sizes " *
                "and the between-group rate can be read from it"
        )
    )
    return spec
end

# `Λ[g, h]`, the rate at which one infective of group `h` contacts one
# member of group `g`: `Λ * I` is then the force of infection on every
# group, for `I` the vector of each group's infectious count, exactly as
# `MixingProcess.rate(from, to)` gives that rate to the exact race.
_contact_matrix(rate, groups) = [rate(h, g) for g in groups, h in groups]

_calendar_value(::Nothing, t::Real) = 1.0
_calendar_value(calendar, t::Real) = calendar_multiplier(calendar, t)

"""
    ImmuneShare(share)

The share of each group that is immune when an approximation for a mixing
population starts, from earlier vaccination or infection: a number between 0
and 1 for every group, or one per group (in the order of the model's
`groups`). They are taken out of the susceptibles at `t = 0`, before the
initial infections are placed. See [`ImmuneCount`](@ref) for a number of
people.
"""
struct ImmuneShare{T}
    share::T
end

"""
    ImmuneCount(count)

The number of people in each group immune when an approximation for a mixing
population starts: one number for every group, or one per group (in the
order of the model's `groups`). See [`ImmuneShare`](@ref) for a share.
"""
struct ImmuneCount{T}
    count::T
end

_per_group(x::Real, G) = fill(Float64(x), G)
function _per_group(x::AbstractVector, G)
    length(x) == G || throw(
        ArgumentError("expected one value per group ($G), got $(length(x))")
    )
    return Float64.(x)
end

function _immune_counts(immune::ImmuneShare, sizes)
    share = _per_group(immune.share, length(sizes))
    all(s -> 0 <= s <= 1, share) ||
        throw(ArgumentError("an ImmuneShare must be between 0 and 1"))
    return share .* sizes
end
function _immune_counts(immune::ImmuneCount, sizes)
    count = _per_group(immune.count, length(sizes))
    all(i -> 0 <= count[i] <= sizes[i], eachindex(sizes)) ||
        throw(ArgumentError("an ImmuneCount must be between 0 and each group's size"))
    return count
end

# The state a run starts from: `n_initial` people infected overall, split
# across the groups in proportion to their size, among the people who are not
# immune.
function _initial_compartments(model, n_initial::Integer, immune)
    1 <= n_initial <= model.population_size ||
        throw(ArgumentError("n_initial must be between 1 and population_size"))
    N = Float64.(model.group_sizes)
    P0 = _immune_counts(immune, N)
    I0 = n_initial .* (N ./ model.population_size)
    all(I0 .<= N .- P0) || throw(
        ArgumentError("the immune people leave too few susceptibles for the initial infections")
    )
    G = length(N)
    return N .- I0 .- P0, I0, zeros(G), P0, zeros(G)
end

# ── Shared integration ───────────────────────────────────────────────

# The state is carried as one vector `x = [S; I; V; P; D]`, each block with
# one entry per group, so the same step serves the mean path of both
# approximations; the linear noise covariance is over the first four blocks.
_unpack(x, G) = (
    view(x, 1:G), view(x, (G + 1):(2G)), view(x, (2G + 1):(3G)),
    view(x, (3G + 1):(4G)), view(x, (4G + 1):(5G)),
)

_outcome(::NoCampaign) = (immune = 0.0, vaccinated = 0.0, hazard = 1.0)
_outcome(c::GroupCampaign) = (immune = c.immune, vaccinated = c.vaccinated, hazard = c.hazard)

# The rate of every reaction in each group: infection of the unvaccinated and
# of the vaccinated susceptibles, removal, and a dose reaching an
# unvaccinated susceptible and leaving them vaccinated or immune. `λ` is each
# group's force of infection, from the contact matrix built when the model
# was constructed, at time `t`; the dose rate is read at `tm`, a time inside
# the step, since the budget is constant over each period the steps land on
# the ends of. A dose reaches a person not yet dosed, so a susceptible's
# share of them is `S` of the `size - D` who are left.
function _mixing_rates(model, x, t, tm)
    G = length(model.groups)
    S, I, V, P, D = _unpack(x, G)
    out = _outcome(model.campaign)
    λ = _calendar_value(model.calendar, t) .* (model.contact_matrix * I)
    r = _dose_rate(model.campaign, model, tm, S, D)
    undosed = max.(Float64.(model.group_sizes) .- D, S, 1.0e-9)
    reach = r ./ undosed .* S
    return (
        infect = S .* λ, infect_vaccinated = out.hazard .* V .* λ,
        remove = model.removal_rate .* I,
        vaccinate = out.vaccinated .* reach, immunise = out.immune .* reach, doses = r,
    )
end

# The per-group drift of `x`.
function _mixing_drift(model, x, t, tm)
    a = _mixing_rates(model, x, t, tm)
    infections = a.infect .+ a.infect_vaccinated
    return vcat(
        -a.infect .- a.vaccinate .- a.immunise,
        infections .- a.remove,
        a.vaccinate .- a.infect_vaccinated,
        a.immunise,
        a.doses
    )
end

# Fourth-order Runge–Kutta for `x`, with the drift evaluated at each of the
# step's four stage times so a calendar scaling the contact matrix is read at
# the time it actually applies to, rather than frozen at the step's start.
function _rk4_mixing_step(model, x, t::Real, h::Real)
    tm = t + h / 2
    k1 = _mixing_drift(model, x, t, tm)
    k2 = _mixing_drift(model, x + h / 2 * k1, t + h / 2, tm)
    k3 = _mixing_drift(model, x + h / 2 * k2, t + h / 2, tm)
    k4 = _mixing_drift(model, x + h * k3, t + h, tm)
    return x + h / 6 * (k1 + 2k2 + 2k3 + k4)
end

# The drift's Jacobian with respect to the four compartment blocks (the
# doses `D` are a given function of time, not noise), by central differences:
# the drift is smooth and low-order in the compartments, so this is as exact as
# a hand-written one would be, and it follows whatever a dose allocation makes
# the rates depend on.
function _mixing_jacobian(model, x, t, tm)
    n = 4 * length(model.groups)
    J = zeros(n, n)
    for k in 1:n
        ε = 1.0e-6 * max(1.0, abs(x[k]))
        up = copy(x)
        up[k] += ε
        down = copy(x)
        down[k] -= ε
        J[:, k] .= (
            _mixing_drift(model, up, t, tm)[1:n] .-
                _mixing_drift(model, down, t, tm)[1:n]
        ) ./ (2ε)
    end
    return J
end

# The Chemical Langevin diffusion for the `4G`-dimensional state
# `(S, I, V, P)`: every reaction moves only its own group's compartments, by
# its stoichiometry, so each adds its rate times the outer product of that
# vector with itself at that group's own place in the matrix. Group `g`'s
# infection reactions are driven by every group `h`'s infectious count, but
# they share group `g`'s stoichiometry, so their propensities sum before
# scaling it rather than needing one outer product per `(g, h)` pair.
const _MIXING_REACTIONS = (
    (:infect, (-1.0, 1.0, 0.0, 0.0)),
    (:infect_vaccinated, (0.0, 1.0, -1.0, 0.0)),
    (:remove, (0.0, -1.0, 0.0, 0.0)),
    (:vaccinate, (-1.0, 0.0, 1.0, 0.0)),
    (:immunise, (-1.0, 0.0, 0.0, 1.0)),
)

function _mixing_diffusion(model, x, t, tm)
    G = length(model.groups)
    a = _mixing_rates(model, x, t, tm)
    Q = zeros(4G, 4G)
    for (name, ν) in _MIXING_REACTIONS
        rate = getproperty(a, name)
        for g in 1:G, (i, νi) in enumerate(ν), (j, νj) in enumerate(ν)
            Q[(i - 1) * G + g, (j - 1) * G + g] += rate[g] * νi * νj
        end
    end
    return Q
end

# The covariance's step over `h` days (explicit Euler, evaluating the
# Jacobian and diffusion at the step's start), and nothing for the mean field.
_covariance_step(::Nothing, model, x, t, h) = nothing
function _covariance_step(Σ::AbstractMatrix, model, x, t, h)
    J = _mixing_jacobian(model, x, t, t + h / 2)
    Q = _mixing_diffusion(model, x, t, t + h / 2)
    Σ = Σ + h * (J * Σ + Σ * J' + Q)
    return (Σ + Σ') / 2   # undo floating-point asymmetry from the update above
end

_cases_by_group(::NoCampaign, model, x) = Float64[]
_cases_by_group(c::GroupCampaign, model, x) = _group_cases(c, model, x)

# A rate of doses must not carry a group past the people it can reach, which
# the step that exhausts them would otherwise overshoot a little.
_cap_doses!(::NoCampaign, model, x) = nothing
function _cap_doses!(c::GroupCampaign, model, x)
    G = length(model.groups)
    D = view(x, (4G + 1):(5G))
    D .= min.(D, c.coverage .* model.group_sizes)
    return nothing
end

# Fixed-step RK4 from `t0` to `max_time`, in steps no longer than
# `model.step` (the last one shortened to land exactly on `max_time`),
# mirroring `_integrate_sir` for the scalar case, and shortened further to land
# on the moments the campaign's doses change. Returns the new `x`, the
# covariance if there was one, and moves `progress` on.
function _integrate_mixing(model, x, progress, Σ, t0, max_time)
    G = length(model.groups)
    campaign = model.campaign
    n_steps = max(1, ceil(Int, (max_time - t0) / model.step))
    dt = (max_time - t0) / n_steps
    t = t0
    _fire_doses!(campaign, model, x, Σ, progress, t)
    for k in 1:n_steps
        t_end = k == n_steps ? max_time : t0 + k * dt
        while t < t_end - 1.0e-12
            h = min(t_end - t, _next_dose_event(campaign, progress, t) - t)
            before = _cases_by_group(campaign, model, x)
            Σ = _covariance_step(Σ, model, x, t, h)
            x = _rk4_mixing_step(model, x, t, h)
            _cap_doses!(campaign, model, x)
            t += h
            _trigger_groups!(
                campaign, progress, before, _cases_by_group(campaign, model, x), t - h, t
            )
            _fire_doses!(campaign, model, x, Σ, progress, t)
        end
    end
    N = Float64.(model.group_sizes)
    S, I, V, P, D = _unpack(x, G)
    S .= clamp.(S, 0.0, N)
    V .= clamp.(V, 0.0, N .- S)
    P .= clamp.(P, 0.0, N .- S .- V)
    I .= clamp.(I, 0.0, N .- S .- V .- P)
    return x, Σ
end

# The campaign's own progress at the start of a run: groups whose expected
# cases are already past the trigger (the initial infections) are triggered at
# `t = 0`.
function _start_progress(model, x)
    progress = _new_progress(model.campaign)
    cases = _cases_by_group(model.campaign, model, x)
    _trigger_groups!(model.campaign, progress, cases, cases, 0.0, 0.0)
    return progress
end

"""
    MixingMeanFieldProcess(spec::ModelSpec; removal_rate, step = 0.05,
                           allocation = TriggeredInPriority(),
                           rng = Random.default_rng())

The deterministic mean-field approximation of a [`MixingProcess`](@ref)
composed with a single exponential removal step, with compartments per mixing
group instead of one set for the whole population:

```
dS_g/dt = -S_g Σ_h Λ[g,h] I_h,  dI_g/dt = S_g Σ_h Λ[g,h] I_h - γ I_g,  dR_g/dt = γ I_g
```

`spec` must be a [`ModelSpec`](@ref) built on a `MixingProcess`, with
`attributes` assigning the keys `mixing_by` groups by. The groups, their
sizes, the between-group rate `Λ[g,h] = rate(h, g)` and any `calendar`
scaling it are all read from `spec`, so the same mixing structure drives the
exact and the approximate dynamics rather than being given twice: the group
sizes are read by building the population `spec` describes (with nobody
infected) and counting who `attributes` puts in each group. The groups must
come out the same on every draw (by `ind.id`, say), since a fixed-size
compartment cannot represent a group whose size is itself random.

`removal_rate` (γ) is the exponential removal rate the exact model's
progression gives its terminal step, exactly as [`MeanFieldProcess`](@ref)'s
own `removal_rate` is. The compartments are integrated as continuous
numbers, not people, by fourth-order Runge–Kutta with a fixed step of
`step` days.

# Vaccination

`spec`'s interventions are read too: a [`GroupVaccination`](@ref), alone or
under [`Scheduled`](@ref) (with `start_time` and/or `end_time`) and
[`CapacityConstrained`](@ref) (a fixed budget, or a schedule such as
[`Steps`](@ref)). Any other intervention is refused, since the approximation
would silently leave it out. The campaign is read from the intervention as it
is written, and nothing is given twice: `group_key` (which must be one of the
`mixing_by` keys; doses go to the groups it names), `efficacy` and `mode`,
`coverage`, `dose_delay`, the window of `Scheduled` and the budget, `period`
and `carry_over` of `CapacityConstrained`. `efficacy`, `coverage` and
`dose_delay` must be numbers, `delay_to_immunity` must be 0 and `waning`
`nothing`, and `on_refusal` must be [`DropRefused`](@ref), which the
compartments do not model otherwise.

Doses go to people whatever their status, so only the share `S_g/(size - D_g)`
of a group's doses, for `D_g` the doses it has had, reaches an unvaccinated
susceptible. What a dose does to one depends on the mode. With
[`AllOrNothingMode`](@ref) a share `efficacy` become immune and the rest
stay exactly as susceptible as before. With [`LeakyMode`](@ref) all become
vaccinated susceptibles, whose force of infection is scaled by
`1 - efficacy`. A mode with no such model is refused.

How a group's discrete trigger and a shared budget become doses is not
settled by the intervention, so it is a choice: `allocation`, a
[`DoseAllocation`](@ref). The default, [`TriggeredInPriority`](@ref), follows
the exact model; [`SplitByGroupSize`](@ref) and
[`SplitBySusceptibles`](@ref) share a budget's daily rate over the groups
instead.

Pass the result to [`simulate`](@ref) for a [`MixingMeanFieldState`](@ref),
and that state back to [`simulate`](@ref), with a later `max_time`, to continue
the integration from where it stopped. See
[`MixingLinearNoiseProcess`](@ref) for the same dynamics with Gaussian noise
around the mean path, and [`HybridProcess`](@ref) for switching between
exact simulation of `spec` and this approximation.

# Example

```julia
using EpiBranch, Distributions
n = [2_000, 8_000]
band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
mixing = MixingProcess(; population_size = sum(n), mixing_by = (:age_band,),
    rate = (from, to) -> (from == to ? 0.3 : 0.1) / n[to[1]])
progression = [Transition(:recovered; rate = 0.2, terminal = true)]
spec = ModelSpec(mixing; progression, attributes = band)
model = MixingMeanFieldProcess(spec; removal_rate = 0.2)
state = simulate(model; n_initial = 5, max_time = 60.0)

# The same model with a vaccination campaign: a group's first case triggers
# doses for its members, from a budget of 100 a day between days 10 and 17.
clinical = clinical_presentation(incubation_period = Dirac(0.0), prob_asymptomatic = 0.0)
campaign = CapacityConstrained(
    GroupVaccination(efficacy = 0.9, eligibility = OnSymptomOnset(), group_key = :age_band);
    budget_per_period = Steps([10.0, 17.0], [0.0, 100.0, 0.0]), period = 1.0)
vaccinated = ModelSpec(mixing; progression, attributes = [band, clinical],
    interventions = [campaign])
model = MixingMeanFieldProcess(vaccinated; removal_rate = 0.2)
state = simulate(model; n_initial = 5, immune = ImmuneShare(0.1), max_time = 60.0)
```
"""
struct MixingMeanFieldProcess{G, C, K}
    population_size::Int
    mixing_by::Tuple
    groups::Vector{G}
    group_sizes::Vector{Int}
    contact_matrix::Matrix{Float64}
    calendar::C
    removal_rate::Float64
    step::Float64
    campaign::K
end

function MixingMeanFieldProcess(
        spec; removal_rate::Real, step::Real = 0.05,
        allocation::DoseAllocation = TriggeredInPriority(),
        rng::AbstractRNG = Random.default_rng()
    )
    spec = _mixing_process_spec(spec, "MixingMeanFieldProcess")
    bare = process(spec)
    (isfinite(removal_rate) && removal_rate >= 0) ||
        throw(ArgumentError("removal_rate must be finite and non-negative"))
    step > 0 || throw(ArgumentError("step must be positive"))
    groups, sizes = _mixing_group_layout(spec, bare.mixing_by, rng)
    Λ = float.(_contact_matrix(bare.rate, groups))
    all(r -> isfinite(r) && r >= 0, Λ) ||
        throw(ArgumentError("mixing rates must be finite and non-negative"))
    campaign = _read_campaign(
        spec, bare.mixing_by, groups, allocation, "MixingMeanFieldProcess"
    )
    return MixingMeanFieldProcess(
        sum(sizes), bare.mixing_by, groups, sizes, Λ, bare.calendar,
        Float64(removal_rate), Float64(step), campaign
    )
end

population_size(m::MixingMeanFieldProcess) = m.population_size

function Base.show(io::IO, m::MixingMeanFieldProcess)
    return print(
        io,
        "MixingMeanFieldProcess(population_size=$(m.population_size), " *
            "groups=$(length(m.groups)), γ=$(m.removal_rate))"
    )
end

"""
    MixingMeanFieldState

The compartment counts of a [`MixingMeanFieldProcess`](@ref) at a point in
time, one continuous count per mixing group, in the same order as `groups`:
`S` the susceptibles who have had no dose, `I` the infectious, `R` the
removed, `V` the susceptibles who have had a dose that has not made them
immune and `P` the immune (from before the outbreak or an all-or-nothing dose
that took), and `D` the doses given, whatever their recipient's status. `t`
is the time (days since the start of the outbreak) the state has been
integrated to, and `campaign` the progress of the vaccination campaign.
"""
struct MixingMeanFieldState{G}
    groups::Vector{G}
    S::Vector{Float64}
    I::Vector{Float64}
    R::Vector{Float64}
    t::Float64
    V::Vector{Float64}
    P::Vector{Float64}
    D::Vector{Float64}
    campaign::CampaignProgress
end

# A state with no vaccination and nobody immune, as the model before
# vaccination was read from a spec had it.
function MixingMeanFieldState(groups, S, I, R, t)
    G = length(groups)
    return MixingMeanFieldState(
        groups, S, I, R, t, zeros(G), zeros(G), zeros(G), _new_progress(NoCampaign())
    )
end

function Base.show(io::IO, s::MixingMeanFieldState)
    return print(
        io,
        "MixingMeanFieldState(S=$(round(sum(s.S), digits = 1)), " *
            "I=$(round(sum(s.I), digits = 1)), R=$(round(sum(s.R), digits = 1)), " *
            "groups=$(length(s.groups)), t=$(s.t))"
    )
end

_pack(S, I, V, P, D) = vcat(S, I, V, P, D)

function _removed(model, x)
    G = length(model.groups)
    S, I, V, P, _ = _unpack(x, G)
    return Float64.(model.group_sizes) .- S .- I .- V .- P
end

"""
    simulate(model::MixingMeanFieldProcess; n_initial = 1,
             immune = ImmuneShare(0.0), max_time)

Integrate the mean-field ODE of `model` from `t = 0`, with `n_initial` people
infected overall, split across the mixing groups in proportion to their
size, up to `max_time`. `immune`, an [`ImmuneShare`](@ref) or an
[`ImmuneCount`](@ref), sets how many people of each group are immune from the
start, taken out of the susceptibles. Returns a [`MixingMeanFieldState`](@ref);
continue it further with `simulate(model, state; max_time)`.
"""
function simulate(
        model::MixingMeanFieldProcess; n_initial::Integer = 1,
        immune = ImmuneShare(0.0), max_time::Real
    )
    x = _pack(_initial_compartments(model, n_initial, immune)...)
    progress = _start_progress(model, x)
    return _mixing_mean_state(model, x, progress, 0.0, max_time)
end

"""
    simulate(model::MixingMeanFieldProcess, state::MixingMeanFieldState; max_time)

Continue integrating `state`'s mean-field ODE up to `max_time`, which must
be later than `state.t`.
"""
function simulate(
        model::MixingMeanFieldProcess, state::MixingMeanFieldState; max_time::Real
    )
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    x = _pack(state.S, state.I, state.V, state.P, state.D)
    return _mixing_mean_state(model, x, _copy(state.campaign), state.t, max_time)
end

function _mixing_mean_state(model, x, progress, t0, max_time)
    x, _ = _integrate_mixing(model, x, progress, nothing, t0, max_time)
    G = length(model.groups)
    S, I, V, P, D = _unpack(x, G)
    return MixingMeanFieldState(
        model.groups, collect(S), collect(I), _removed(model, x), Float64(max_time),
        collect(V), collect(P), collect(D), progress
    )
end

"""
    MixingLinearNoiseProcess(spec::ModelSpec; removal_rate, step = 0.05,
                             allocation = TriggeredInPriority(),
                             rng = Random.default_rng())

The same multi-group dynamics as [`MixingMeanFieldProcess`](@ref),
represented as a Gaussian process around the deterministic mean path (the
linear noise approximation): the state holds the mean compartment counts and
their covariance, which grows from the demographic noise of the
infection, removal and vaccination reactions in every group and is damped by
the linearised dynamics around the mean.

Accepts the same keywords as [`MixingMeanFieldProcess`](@ref), built from the
same `ModelSpec` (including its vaccination campaign), and the mean path
integrates identically; only the covariance is new. The number of doses
given is taken as known, so the noise a campaign adds is that of which
people they reach: who among a group's susceptibles is dosed, and whom the
dose protects.

# Example

```julia
using EpiBranch, Distributions
n = [2_000, 8_000]
band = (rng, ind) -> (ind.state[:age_band] = ind.id <= n[1] ? 1 : 2)
spec = ModelSpec(
    MixingProcess(; population_size = sum(n), mixing_by = (:age_band,),
        rate = (from, to) -> (from == to ? 0.3 : 0.1) / n[to[1]]);
    progression = [Transition(:recovered; rate = 0.2, terminal = true)],
    attributes = band)
model = MixingLinearNoiseProcess(spec; removal_rate = 0.2)
state = simulate(model; n_initial = 5, max_time = 60.0)
state.covariance   # 4|groups|×4|groups| covariance of (S_1, ..., I_1, ..., V_1, ..., P_1, ...)
```
"""
struct MixingLinearNoiseProcess{G, C, K}
    population_size::Int
    mixing_by::Tuple
    groups::Vector{G}
    group_sizes::Vector{Int}
    contact_matrix::Matrix{Float64}
    calendar::C
    removal_rate::Float64
    step::Float64
    campaign::K
end

function MixingLinearNoiseProcess(
        spec; removal_rate::Real, step::Real = 0.05,
        allocation::DoseAllocation = TriggeredInPriority(),
        rng::AbstractRNG = Random.default_rng()
    )
    mean_field = MixingMeanFieldProcess(spec; removal_rate, step, allocation, rng)
    return MixingLinearNoiseProcess(
        mean_field.population_size, mean_field.mixing_by, mean_field.groups,
        mean_field.group_sizes, mean_field.contact_matrix, mean_field.calendar,
        mean_field.removal_rate, mean_field.step, mean_field.campaign
    )
end

population_size(m::MixingLinearNoiseProcess) = m.population_size

function Base.show(io::IO, m::MixingLinearNoiseProcess)
    return print(
        io,
        "MixingLinearNoiseProcess(population_size=$(m.population_size), " *
            "groups=$(length(m.groups)), γ=$(m.removal_rate))"
    )
end

"""
    MixingLinearNoiseState

The Gaussian state of a [`MixingLinearNoiseProcess`](@ref): mean compartment
counts `S`, `I`, `R`, `V`, `P` and `D` (one per mixing group, as for
[`MixingMeanFieldState`](@ref)), time `t`, the progress `campaign` of the
vaccination campaign, and `covariance`, the `4|groups| × 4|groups|`
covariance of `(S_1, ..., S_g, I_1, ..., I_g, V_1, ..., V_g, P_1, ..., P_g)`
(`R` has no separate uncertainty, since each group's is what is left of its
size, and the doses are taken as known). Without vaccination `V` and `P`
stay at what they start as, and the first `2|groups|` rows and columns are
the covariance of `S` and `I` alone.
"""
struct MixingLinearNoiseState{G}
    groups::Vector{G}
    S::Vector{Float64}
    I::Vector{Float64}
    R::Vector{Float64}
    t::Float64
    covariance::Matrix{Float64}
    V::Vector{Float64}
    P::Vector{Float64}
    D::Vector{Float64}
    campaign::CampaignProgress
end

# A state with no vaccination and nobody immune, from a covariance over just
# `S` and `I` (padded with zeros for the other compartments) or over all four.
function MixingLinearNoiseState(groups, S, I, R, t, covariance)
    G = length(groups)
    Σ = zeros(4G, 4G)
    k = size(covariance, 1)
    Σ[1:k, 1:k] .= covariance
    return MixingLinearNoiseState(
        groups, S, I, R, t, Σ, zeros(G), zeros(G), zeros(G), _new_progress(NoCampaign())
    )
end

function Base.show(io::IO, s::MixingLinearNoiseState)
    return print(
        io,
        "MixingLinearNoiseState(S=$(round(sum(s.S), digits = 1)), " *
            "I=$(round(sum(s.I), digits = 1)), R=$(round(sum(s.R), digits = 1)), " *
            "groups=$(length(s.groups)), t=$(s.t))"
    )
end

"""
    simulate(model::MixingLinearNoiseProcess; n_initial = 1,
             immune = ImmuneShare(0.0), max_time)

Integrate the linear noise approximation of `model` from `t = 0`, with
`n_initial` people infected overall, split across the mixing groups in
proportion to their size, and `immune` people of each group immune (an
[`ImmuneShare`](@ref) or [`ImmuneCount`](@ref)), with no uncertainty yet (a
zero covariance), up to `max_time`. Returns a [`MixingLinearNoiseState`](@ref);
continue it further with `simulate(model, state; max_time)`.
"""
function simulate(
        model::MixingLinearNoiseProcess; n_initial::Integer = 1,
        immune = ImmuneShare(0.0), max_time::Real
    )
    x = _pack(_initial_compartments(model, n_initial, immune)...)
    progress = _start_progress(model, x)
    G = length(model.groups)
    return _mixing_lna_state(model, x, progress, zeros(4G, 4G), 0.0, max_time)
end

"""
    simulate(model::MixingLinearNoiseProcess, state::MixingLinearNoiseState; max_time)

Continue `state`'s mean path and covariance up to `max_time`, which must be
later than `state.t`.
"""
function simulate(
        model::MixingLinearNoiseProcess, state::MixingLinearNoiseState; max_time::Real
    )
    max_time > state.t || throw(
        ArgumentError(
            "max_time ($max_time) must be past the state's own ($(state.t)) to continue it"
        )
    )
    x = _pack(state.S, state.I, state.V, state.P, state.D)
    return _mixing_lna_state(
        model, x, _copy(state.campaign), copy(state.covariance), state.t, max_time
    )
end

function _mixing_lna_state(model, x, progress, Σ, t0, max_time)
    x, Σ = _integrate_mixing(model, x, progress, Σ, t0, max_time)
    G = length(model.groups)
    S, I, V, P, D = _unpack(x, G)
    return MixingLinearNoiseState(
        model.groups, collect(S), collect(I), _removed(model, x), Float64(max_time), Σ,
        collect(V), collect(P), collect(D), progress
    )
end
