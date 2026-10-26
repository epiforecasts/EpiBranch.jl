# A custom terminal transition with a `probability` field that gates nothing:
# the transition always occurs, and `terminal_event` reads the field only to
# label the outcome. Defined at module scope
# because struct definitions can't live inside @testset. Used below to check
# that `terminal_certainty` is read by dispatch rather than guessed from
# field layout.
struct AlwaysOccursRule <: AbstractClinicalTransition
    probability::Float64
end
EpiBranch.is_terminal(::AlwaysOccursRule) = true
function EpiBranch.terminal_event(r::AlwaysOccursRule, individual)
    return (individual.infection_time + 1.0, r.probability > 0.5 ? :died : :recovered)
end

# Two interventions meant to block every transmission, each written so that
# the engine never reaches the method: `BlockWrongArity`'s `competing_risk`
# is missing its `state` argument, and `BlockUnqualified`'s is defined
# without the `EpiBranch.` prefix, which gives this module a new function of
# the same name rather than a method on the package's own. Both therefore
# fall back to the no-op default. Defined at module scope for the same
# reason as `AlwaysOccursRule` above.
struct BlockWrongArity <: AbstractIntervention end
EpiBranch.competing_risk(::BlockWrongArity, parent, contact) = Risk(block_probability = 1.0)

# The same mistake on a second type, for the wrapper check: a type is
# examined once per session, so re-checking `BlockWrongArity` inside a
# wrapper would find it already seen.
struct WrappedWrongArity <: AbstractIntervention end
EpiBranch.competing_risk(::WrappedWrongArity, parent, contact) =
    Risk(block_probability = 1.0)

struct BlockUnqualified <: AbstractIntervention end
competing_risk(::BlockUnqualified, parent, contact, state) = Risk(block_probability = 1.0)

# One hook written correctly and one written without the `EpiBranch.` prefix,
# so the second lands on the test module's own `competing_risk` above. The
# intervention is reachable through the first, so only a per-hook check sees
# the second.
struct MixedPrefix <: AbstractIntervention end
EpiBranch.initialise_individual!(::MixedPrefix, ind, state) =
    (ind.state[:flagged] = true; nothing)
competing_risk(::MixedPrefix, parent, contact, state) = Risk(block_probability = 1.0)

# A correct intervention written in a module that also holds the author's own
# helper of a hook's name. The helper takes an intervention first but at an
# arity no call uses, so it is not a mis-written hook and must not be
# reported as one: writing it as `EpiBranch.trace_contacts!` would break it.
module _OwnHelpers
    using EpiBranch
    using EpiBranch: AbstractIntervention, Risk
    struct HasHelper <: AbstractIntervention
        traced::Vector{Int}
    end
    EpiBranch.competing_risk(::HasHelper, parent, contact, state) =
        Risk(block_probability = 1.0)
    trace_contacts!(h::HasHelper, ids) = append!(h.traced, ids)
    keep_active(a, b) = nothing
    struct NoHooks <: AbstractIntervention end
end

# Acts through no hook the package knows of, standing for one that reaches the
# model through a seam of its own package, and says so through `has_effect`.
struct _OwnSeam <: AbstractIntervention end
EpiBranch.has_effect(::_OwnSeam) = true

# Never passed to `ModelSpec`, so the cache's own behaviour can be read
# without another construction having examined the type first.
struct _CacheProbe <: AbstractIntervention end

# Implements only the least common of `_effect_hooks()`, to check the full set
# is read rather than a subset headlined by `competing_risk`.
struct _ActionsHookOnly <: AbstractIntervention end
EpiBranch.intervention_actions(::_ActionsHookOnly, state, candidates) = nothing

# A `competing_risk` hook written with a trailing `args...`: a method a
# four-argument call does reach, despite `methods` listing its own arity as
# two rather than four.
struct BlockVariadic <: AbstractIntervention end
EpiBranch.competing_risk(::BlockVariadic, args...) = Risk(block_probability = 1.0)

# The same hook with the vararg's length named: `Vararg{Any, N} where {N}`
# has an `N` that is a type variable rather than a count.
struct BlockVariadicN <: AbstractIntervention end
EpiBranch.competing_risk(::BlockVariadicN, args::Vararg{Any, N}) where {N} =
    Risk(block_probability = 1.0)

# An intervention whose only declared effect is on the likelihood, as the
# extending guide's `Prophylaxis` is.
struct _SusceptibilityOnly <: AbstractIntervention end
EpiBranch.susceptibility_components(::_SusceptibilityOnly, host) =
    (1 => EpiBranch.HazardScaling(2.0, 0.5),)

# One hook written correctly and one with an argument dropped: the shape a
# real intervention takes when a hook is mistyped, where the intervention
# itself is plainly not inert.
struct BlockOneGoodOneShort <: AbstractIntervention end
EpiBranch.initialise_individual!(::BlockOneGoodOneShort, ind, state) = nothing
EpiBranch.competing_risk(::BlockOneGoodOneShort, parent, contact) =
    Risk(block_probability = 1.0)

# A hook written with every argument variadic, which names no intervention
# type at all. Defined on a function of its own rather than on one of the
# package's, so that the rest of the suite does not inherit it.
_all_variadic(args...) = nothing

# A family of interventions whose hook is written once on the abstract parent,
# with a child whose override drops an argument. The parent's method answers
# the call, so the child's total block never applies: the check has to ask per
# method, since asking per hook finds the parent's and falls silent.
abstract type _Family <: AbstractIntervention end
EpiBranch.competing_risk(::_Family, parent, contact, state) =
    Risk(block_probability = 0.0)
struct _FamilyShort <: _Family end
EpiBranch.competing_risk(::_FamilyShort, parent, contact) =
    Risk(block_probability = 1.0)

# The other way round: the short method is the parent's and the child's is
# correct, so the child reaches its own and nothing is wrong with it.
abstract type _FamilyB <: AbstractIntervention end
EpiBranch.competing_risk(::_FamilyB, parent, contact) = Risk(block_probability = 1.0)
struct _FamilyBOK <: _FamilyB end
EpiBranch.competing_risk(::_FamilyBOK, parent, contact, state) =
    Risk(block_probability = 1.0)

# A correct hook alongside a convenience overload of it for the same type: the
# shorter one is unreachable from the engine, but the type's own hook answers
# the call, so there is nothing to report.
struct _ExtraOverload <: AbstractIntervention end
EpiBranch.competing_risk(::_ExtraOverload, parent, contact, state) =
    Risk(block_probability = 1.0)
EpiBranch.competing_risk(::_ExtraOverload, contact) = Risk(block_probability = 1.0)

# A family whose hook is written once on the abstract parent, with a child
# whose own override is written without the `EpiBranch.` prefix. The parent's
# method answers the call and the child's total block never applies, so asking
# whether the hook has an own method at all finds the parent's and falls
# silent: the question has to be whether a method at least as specific as the
# stray one answers.
abstract type _FamilyC <: AbstractIntervention end
EpiBranch.competing_risk(::_FamilyC, parent, contact, state) =
    Risk(block_probability = 0.0)
struct _FamilyUnqualified <: _FamilyC end
competing_risk(::_FamilyUnqualified, parent, contact, state) =
    Risk(block_probability = 1.0)

# The same family with the child's override written correctly.
struct _FamilyQualified <: _FamilyC end
EpiBranch.competing_risk(::_FamilyQualified, parent, contact, state) =
    Risk(block_probability = 1.0)

# A stray unprefixed method alongside a correct one for the same type: the
# type's own hook answers the call, so the stray function is a leftover rather
# than a mistake to report.
struct _BothPrefixes <: AbstractIntervention end
competing_risk(::_BothPrefixes, parent, contact, state) = Risk(block_probability = 1.0)
EpiBranch.competing_risk(::_BothPrefixes, parent, contact, state) =
    Risk(block_probability = 1.0)

# A hook shared by two intervention types through a `Union` first parameter,
# which answers a call for either of them. `nameof` has no method for a
# `Union`, and `<:` makes a union a subtype of neither member, so both the
# naming and the specificity questions have to cope with one. Here with a
# convenience overload alongside, as `_ExtraOverload` has, so there is nothing
# to report and nothing to name.
struct _UnionShare <: AbstractIntervention end
struct _UnionShareB <: AbstractIntervention end
EpiBranch.competing_risk(::Union{_UnionShare, _UnionShareB}, parent, contact, state) =
    Risk(block_probability = 0.5)
EpiBranch.competing_risk(::_UnionShare, contact) = Risk(block_probability = 0.5)

# `BlockWrongArity`'s mistake written once for two types: there is something to
# report, and the union is what it has to be reported against.
struct _UnionShort <: AbstractIntervention end
struct _UnionShortB <: AbstractIntervention end
EpiBranch.competing_risk(::Union{_UnionShort, _UnionShortB}, parent, contact) =
    Risk(block_probability = 1.0)

# A correct hook written for a union, with an unprefixed leftover for one of
# its members: the union covers that member, so the leftover is a leftover.
struct _UnionBoth <: AbstractIntervention end
struct _UnionBothB <: AbstractIntervention end
EpiBranch.competing_risk(::Union{_UnionBoth, _UnionBothB}, parent, contact, state) =
    Risk(block_probability = 1.0)
competing_risk(::_UnionBoth, parent, contact, state) = Risk(block_probability = 1.0)

# A child whose override drops an argument, under both a parent hook and a
# wider union hook that covers the parent as well. The engine reaches the
# parent's method, so that is the one to name, and which of the two the method
# list offers first differs between Julia versions, so the choice cannot rest
# on that order.
abstract type _RankFam <: AbstractIntervention end
struct _RankKid <: _RankFam end
struct _RankOther <: AbstractIntervention end
EpiBranch.competing_risk(::Union{_RankFam, _RankOther}, parent, contact, state) =
    Risk(block_probability = 0.25)
EpiBranch.competing_risk(::_RankFam, parent, contact, state) =
    Risk(block_probability = 0.0)
EpiBranch.competing_risk(::_RankKid, parent, contact) = Risk(block_probability = 1.0)

# A parametric intervention family, which is how the package's own
# interventions are written, with the dropped-argument method on one
# instantiation and the live one on the family. The two differ only in the
# parameters, so a name without them would read as the same type twice.
struct _ParamIv{D} <: AbstractIntervention
    delay::D
end
EpiBranch.competing_risk(::_ParamIv, parent, contact, state) =
    Risk(block_probability = 0.1)
EpiBranch.competing_risk(::_ParamIv{Float64}, parent, contact) =
    Risk(block_probability = 1.0)

# A parametric family whose only hook is written for another instantiation, so
# the composed value reaches none of its own methods, and one whose stray
# unprefixed method is on a different instantiation from its correct one. Both
# messages have to name the instantiation to say anything true.
struct _ParamInert{D} <: AbstractIntervention
    delay::D
end
EpiBranch.competing_risk(::_ParamInert{Float64}, parent, contact, state) =
    Risk(block_probability = 1.0)

struct _ParamStray{D} <: AbstractIntervention
    delay::D
end
EpiBranch.competing_risk(::_ParamStray{Int}, parent, contact, state) =
    Risk(block_probability = 1.0)
competing_risk(::_ParamStray{Float64}, parent, contact, state) =
    Risk(block_probability = 1.0)

# The same two mistakes with the hooks written `where {D}`, which leaves each
# method's first parameter holding a type variable of its own.
struct _WhereStray{D} <: AbstractIntervention
    delay::D
end
EpiBranch.initialise_individual!(::_WhereStray{D}, ind, state) where {D} = nothing
competing_risk(::_WhereStray{D}, parent, contact, state) where {D} =
    Risk(block_probability = 1.0)

struct _WhereOverload{D} <: AbstractIntervention
    delay::D
end
EpiBranch.competing_risk(::_WhereOverload{D}, parent, contact, state) where {D} =
    Risk(block_probability = 1.0)
EpiBranch.competing_risk(::_WhereOverload{D}, parent, contact) where {D} =
    Risk(block_probability = 1.0)

# An intervention family in a base module with its concrete types in a
# sibling, which is how the extending guide teaches subtyping: the stray
# method is written where the family is, not where the type is, so the module
# to report is the method's own.
module _StrayHome
    using EpiBranch
    using EpiBranch: AbstractIntervention, Risk
    export competing_risk
    abstract type _StrayFam <: AbstractIntervention end
    competing_risk(::_StrayFam, parent, contact, state) =
        Risk(block_probability = 1.0)
end
module _StrayUser
    using EpiBranch
    using .._StrayHome
    using .._StrayHome: _StrayFam
    struct _StrayKid <: _StrayFam end
end

# A hook whose default is not a no-op: `analytic_offspring_effect` throws, so
# the warning cannot claim the engine carries on without the intervention.
# Written at four arguments, one more than either arity the engine calls it
# with, so it stays a mis-written hook rather than a legitimate three-argument
# override.
struct _AnalyticOnly <: AbstractIntervention end
EpiBranch.initialise_individual!(::_AnalyticOnly, ind, state) = nothing
EpiBranch.analytic_offspring_effect(::_AnalyticOnly, offspring, state, extra) = offspring

# A correct intervention in a module that also holds a variadic helper of a
# hook's name, and an intervention in the same module with no hooks at all.
# The helper is reachable at every arity, so an arity test alone cannot tell
# it from a hook; writing it as `EpiBranch.trace_contacts!` would have it
# answer the engine's four-argument call with `state` where it wants ids.
module _VariadicHelpers
    using EpiBranch
    using EpiBranch: AbstractIntervention, Risk
    struct HasVariadicHelper <: AbstractIntervention
        traced::Vector{Int}
    end
    EpiBranch.competing_risk(::HasVariadicHelper, parent, contact, state) =
        Risk(block_probability = 1.0)
    trace_contacts!(h::HasVariadicHelper, args...) = append!(h.traced, args[1])
    struct OnlyVariadicHelper <: AbstractIntervention end
    keep_active(iv::OnlyVariadicHelper, args...) = nothing
end

# A minimal intervention declaring a closed-form effect on the offspring law
# (halving R), used below to check that the analytical functions on a
# ModelSpec apply a declared effect instead of refusing. Defined at module
# scope for the same reason as AlwaysOccursRule above.
struct _HalfThinning <: AbstractIntervention end
EpiBranch.analytic_offspring_effect(::_HalfThinning, d::NegativeBinomial) =
    NegBin(mean(d) / 2, d.r)

# A process that stores its own interventions, as a custom process may, to
# check the analytical functions apply those as well.
struct _ProcessWithInterventions <: EpiBranch.TransmissionModel
    offspring::NegativeBinomial{Float64}
    interventions::Vector{AbstractIntervention}
end
EpiBranch.single_type_offspring(m::_ProcessWithInterventions) = m.offspring
EpiBranch.interventions(m::_ProcessWithInterventions) = m.interventions

@testset "ModelSpec" begin
    # A ModelSpec composes the modelling layers (progression, interventions,
    # attributes, observation) around a pure transmission process. The process
    # carries none of them; the spec is the single place they are attached.

    @testset "wraps a process faithfully" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        s1 = simulate(bp; max_cases = 500, rng = StableRNG(1))
        s2 = simulate(ModelSpec(bp); max_cases = 500, rng = StableRNG(1))
        @test s2.cumulative_cases == s1.cumulative_cases
    end

    @testset "layers on the spec are applied" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        attr = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))

        spec = ModelSpec(bp; interventions = [iso], attributes = attr)
        @test EpiBranch.interventions(spec) == [iso]
        @test EpiBranch.attributes(spec) === attr

        # The isolation layer takes effect: some cases are isolated.
        state = simulate(spec; max_cases = 500, rng = StableRNG(7))
        @test count(is_isolated, state.individuals) > 0
    end

    @testset "keywords set the layers; default to none" begin
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        bp = BranchingProcess(Poisson(2.0), Exponential(5.0))
        # A bare process carries no layers, so the spec defaults to none.
        @test isempty(EpiBranch.interventions(ModelSpec(bp)))
        # ...and a keyword attaches one.
        @test length(EpiBranch.interventions(ModelSpec(bp; interventions = [iso]))) == 1
    end

    @testset "progression and process are public, and a spec rebuilds from itself" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        prog = [Transition(:onset; from = :infection, delay = Exponential(3.0))]
        iso = Isolation(onset_to_isolation_delay = Exponential(1.0), duration = Inf)
        attr = clinical_presentation(incubation_period = LogNormal(1.5, 0.5))
        spec = ModelSpec(
            bp; progression = prog, interventions = [iso], attributes = attr
        )

        if VERSION >= v"1.11"
            @test Base.ispublic(EpiBranch, :progression)
            @test Base.ispublic(EpiBranch, :process)
        else
            @info "Skipping Base.ispublic check on Julia $VERSION"
            @test_skip true
        end
        @test EpiBranch.progression(spec) == prog
        @test EpiBranch.process(spec) === bp

        # Every keyword left out keeps that layer of `spec`.
        extra = demographics(age_distribution = Normal(50, 20))
        rebuilt = ModelSpec(spec; attributes = [EpiBranch.attributes(spec); extra])
        @test EpiBranch.progression(rebuilt) == prog
        @test EpiBranch.interventions(rebuilt) == [iso]
        @test EpiBranch.attributes(rebuilt) == [attr, extra]
        @test EpiBranch.process(rebuilt) === bp
    end

    @testset "first-class across the analytical and distribution API" begin
        bp = BranchingProcess(NegativeBinomial(2.5, 0.5), Exponential(5.0))
        spec = ModelSpec(bp)
        # structural accessors delegate to the wrapped process
        @test EpiBranch.n_types(spec) == 1
        @test EpiBranch.population_size(spec) == EpiBranch.population_size(bp)
        @test single_type_offspring(spec) == single_type_offspring(bp)
        # offspring-based analytical helpers accept a spec
        @test extinction_probability(spec) == extinction_probability(bp)
        @test epidemic_probability(spec) == epidemic_probability(bp)
        @test probability_contain(spec) == probability_contain(bp)
        @test proportion_transmission(spec) == proportion_transmission(bp)
        @test proportion_cluster_size(spec) == proportion_cluster_size(bp)
        # end-of-outbreak needs a single-window, no-observation model
        bpg = BranchingProcess(NegativeBinomial(2.5, 0.5), LogNormal(1.6, 0.5))
        @test end_of_outbreak_probability(ModelSpec(bpg), 10.0) ==
            end_of_outbreak_probability(bpg, 10.0)
        # distribution entry points
        @test offspring_distribution(spec) == offspring_distribution(bp)
        @test pdf(chain_size_distribution(spec), 2) ≈ pdf(chain_size_distribution(bp), 2)
        subc = BranchingProcess(Poisson(0.6))
        @test logpdf(chain_length_distribution(ModelSpec(subc)), [0, 1, 0]) ≈
            logpdf(chain_length_distribution(subc), [0, 1, 0])
        # loglikelihood through a spec: analytical chain sizes and chain lengths
        @test loglikelihood(ChainSizes([1, 2, 1, 3]), spec) ≈
            loglikelihood(ChainSizes([1, 2, 1, 3]), bp)
        @test loglikelihood(ChainLengths([0, 1, 0]), ModelSpec(subc)) ≈
            loglikelihood(ChainLengths([0, 1, 0]), subc)
        # batch simulation through a spec
        @test length(simulate(spec, 20; max_cases = 100, rng = StableRNG(3))) == 20
    end

    @testset "analytical functions refuse to ignore interventions" begin
        # An intervention without a declared closed-form offspring effect (most
        # built-in ones: contact tracing's effect depends on who it reaches and
        # when, which has no closed form here) must make every analytical
        # function on the spec refuse, rather than silently answer as if it
        # were not there. Isolation is not one of these any more — see
        # `test_analytical.jl` for its own closed form and the limits of it.
        bp = BranchingProcess(NegativeBinomial(2.5, 0.5), Gamma(4.0, 1.25))
        ct = ContactTracing(
            probability = 0.8, isolation_to_trace_delay = Exponential(1.0),
            action = Quarantine(duration = 7.0)
        )
        spec = ModelSpec(bp; interventions = [ct])

        for f in (
                extinction_probability, epidemic_probability, probability_contain,
                reproduction_number, proportion_transmission, offspring_distribution,
                proportion_cases_individual, proportion_cases_offspring,
                proportion_cluster_size, spec -> end_of_outbreak_probability(spec, 7.0),
            )
            @test_throws r"ContactTracing" f(spec)
            @test_throws r"from simulated outbreaks, simulate\(spec, n\)" f(spec)
            @test_throws r"probability_contain\(R, k; ind_control, pop_control\)" f(spec)
            @test_throws r"containment_probability\(spec, n\)" f(spec)
        end

        # A spec with no interventions is unaffected.
        bare = ModelSpec(bp)
        @test extinction_probability(bare) == extinction_probability(bp)
    end

    @testset "a process's own interventions are applied too" begin
        iso = Isolation(onset_to_isolation_delay = 0.0, duration = Inf)
        isolated = _ProcessWithInterventions(NegBin(2.5, 0.5), [iso])
        for f in (
                extinction_probability, epidemic_probability, probability_contain,
                reproduction_number, proportion_transmission, offspring_distribution,
                proportion_cases_individual, proportion_cases_offspring,
                proportion_cluster_size,
            )
            @test_throws r"Isolation" f(isolated)
        end

        thinned = _ProcessWithInterventions(NegBin(2.5, 0.5), [_HalfThinning()])
        @test offspring_distribution(thinned) == NegBin(1.25, 0.5)
        @test reproduction_number(thinned) ≈ 1.25
    end

    @testset "a declared offspring effect is applied analytically" begin
        # `_HalfThinning` (defined at module scope above) declares a closed-form
        # effect, so the analytical functions apply it instead of refusing.
        bp = BranchingProcess(NegativeBinomial(2.5, 0.5))
        spec = ModelSpec(bp; interventions = [_HalfThinning()])
        halved = NegBin(reproduction_number(bp) / 2, single_type_offspring(bp).r)

        @test offspring_distribution(spec) == halved
        @test reproduction_number(spec) == reproduction_number(halved)
        @test extinction_probability(spec) == extinction_probability(halved)
        @test epidemic_probability(spec) == epidemic_probability(halved)
        @test probability_contain(spec) == probability_contain(halved)
        @test proportion_transmission(spec) == proportion_transmission(halved)
        @test proportion_cases_individual(spec) == proportion_cases_individual(halved)
        @test proportion_cases_offspring(spec) == proportion_cases_offspring(halved)
        @test proportion_cluster_size(spec) == proportion_cluster_size(halved)
    end

    @testset "analytical chain likelihood ignores non-transmission layers" begin
        # Chain size/length depend only on the offspring law, so composing a
        # progression and attributes (which don't alter the offspring) must
        # leave the analytical likelihood unchanged — only interventions, which
        # thin transmission, route the evaluation through simulation instead.
        bp = BranchingProcess(Poisson(0.8))
        data = ChainSizes([1, 1, 2, 1, 3, 1, 2, 1, 1, 5])
        composed = ModelSpec(
            bp;
            progression = [
                Transition(:onset; from = :infection, delay = Exponential(3.0)),
                Transition(
                    :recovered; from = :infection, delay = Exponential(5.0),
                    terminal = true
                ),
            ],
            attributes = (age = (rng, ind) -> rand(rng) < 0.5 ? :young : :old,)
        )
        @test loglikelihood(data, composed) == loglikelihood(data, ModelSpec(bp))

        # A structured (depleting) model has no single-type offspring, so the
        # analytical path can't be taken — `single_type_offspring` throws and the
        # evaluation falls through to simulation.
        @test_throws ArgumentError single_type_offspring(
            HomogeneousProcess(; transmission_rate = 1.0, population_size = 100)
        )
    end

    @testset "incomplete terminal coverage warns" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        # Two terminal transitions gated independently at p and 1 - p: neither
        # is unconditional, so about p(1 - p) of cases clear both gates'
        # failure and reach no terminal state.
        non_exclusive = [
            Transition(
                :recovered; from = :infection, delay = Exponential(1.0),
                probability = 0.36, terminal = true
            ),
            Transition(
                :died; from = :infection, delay = Exponential(1.0),
                probability = 0.64, terminal = true
            ),
        ]
        @test_logs (:warn, r"gated below.*probability 1") match_mode = :any ModelSpec(
            bp; progression = non_exclusive
        )

        # `exclusive_probabilities` shares one draw, so each gate's own
        # probability is a `Function` — unknowable without an individual, so
        # the check is silently skipped rather than risk a false warning.
        died_p, recovered_p = exclusive_probabilities([0.64, 0.36])
        exclusive = [
            Transition(
                :recovered; from = :infection, delay = Exponential(1.0),
                probability = recovered_p, terminal = true
            ),
            Transition(
                :died; from = :infection, delay = Exponential(1.0),
                probability = died_p, terminal = true
            ),
        ]
        @test_logs ModelSpec(bp; progression = exclusive)

        # `Recovery` has no `probability` field at all — always occurs once its
        # anchor is reached — so pairing it with a gated `Death` guarantees
        # coverage and stays silent.
        @test_logs ModelSpec(
            bp;
            progression = [
                Recovery(delay = Exponential(1.0)),
                Death(delay = Exponential(1.0), probability = 0.3),
            ]
        )

        # A single terminal transition at the default probability (1.0) is
        # unconditional on its own.
        @test_logs ModelSpec(
            bp;
            progression = [
                Transition(
                    :recovered; from = :infection,
                    delay = Exponential(1.0), terminal = true
                ),
            ]
        )

        # A single terminal transition gated below 1, with nothing else to
        # cover the shortfall, warns even alone.
        @test_logs (:warn, r"gated below.*probability 1") match_mode = :any ModelSpec(
            bp; progression = [Death(delay = Exponential(1.0), probability = 0.3)]
        )

        # A `Function`-valued probability cannot be judged statically, so it
        # is not flagged even though it might leave some cases uncovered.
        @test_logs ModelSpec(
            bp;
            progression = [
                Death(
                    delay = Exponential(1.0),
                    probability = (rng, ind) -> 0.3
                ),
            ]
        )

        # No terminal transitions at all is not the gap this check catches.
        @test_logs ModelSpec(
            bp;
            progression = [Transition(:onset; from = :infection, delay = 1.0)]
        )

        # `terminal_certainty` is dispatched, so a `probability` field that
        # gates nothing cannot mislead it: `AlwaysOccursRule` has one, and the
        # transition always occurs. Left at the default (unknown), the
        # transition is skipped rather than misjudged from its field layout.
        @test ismissing(EpiBranch.terminal_certainty(AlwaysOccursRule(0.0)))
        @test_logs ModelSpec(bp; progression = [AlwaysOccursRule(0.0)])
    end

    @testset "an intervention with no reachable hook warns" begin
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))

        # Wrong arity: the method exists but is never the one a four-argument
        # call to `competing_risk` resolves to. The hook at fault is named,
        # rather than every hook being listed.
        @test_logs (
            :warn,
            r"BlockWrongArity has a method of its own for competing_risk, but at no number of arguments the engine calls it with, so that method is never reached: the engine uses the package's own default for competing_risk instead",
        ) match_mode = :any ModelSpec(bp; interventions = [BlockWrongArity()])
        # Unqualified: the method belongs to a different function entirely,
        # which is found in the module the intervention was written in.
        @test_logs (
            :warn,
            r"defines a method of competing_risk for BlockUnqualified, but on its own function",
        ) match_mode = :any ModelSpec(bp; interventions = [BlockUnqualified()])
        # A missing prefix on one hook of an intervention whose other hooks
        # are fine: the intervention is reachable, so only the per-hook
        # check can see it.
        @test_logs (
            :warn,
            r"defines a method of competing_risk for MixedPrefix, but on its own function",
        ) match_mode = :any ModelSpec(bp; interventions = [MixedPrefix()])
        # Wrapping does not hide it: the check reads the wrapped type, since
        # a wrapper such as `Scheduled` always has its own delegating methods
        # whatever it wraps.
        @test_logs (
            :warn,
            r"WrappedWrongArity has a method of its own for competing_risk",
        ) match_mode = :any ModelSpec(
            bp; interventions = [Scheduled(WrappedWrongArity(); start_time = 1.0)]
        )
        # A type is examined once for a given state of the method tables, and
        # defining any method advances the world counter, which arms the
        # check again: a hook rewritten in the REPL is examined afresh.
        @test EpiBranch._first_sight(_CacheProbe)
        @test !EpiBranch._first_sight(_CacheProbe)
        @eval _world_bump() = nothing
        @test EpiBranch._first_sight(_CacheProbe)
        # The author's own helper of a hook's name is not a mis-written hook,
        # and an intervention in that module with no hooks at all still gets
        # the general message rather than a diagnosis of the helper.
        @test_logs ModelSpec(
            bp; interventions = [_OwnHelpers.HasHelper(Int[])]
        )
        @test_logs (
            :warn,
            r"NoHooks has a method of its own for none of the hooks.*`EpiBranch\.has_effect\(::NoHooks\) = true`"s,
        ) match_mode = :any ModelSpec(bp; interventions = [_OwnHelpers.NoHooks()])
        # A hook outside `competing_risk`, at the right arity, is enough to
        # stay silent: the check reads the full set, not just `competing_risk`.
        @test_logs ModelSpec(bp; interventions = [_ActionsHookOnly()])
        # A trailing `args...` is reachable at the call's arity even though
        # `methods` reports its own arity as something other than `n`.
        @test_logs ModelSpec(bp; interventions = [BlockVariadic()])
        # Naming the vararg's length leaves a type variable where a count
        # would be, which the check must not try to add to.
        @test_logs ModelSpec(bp; interventions = [BlockVariadicN()])
        # A hook at an arity the engine never calls is reported even when the
        # intervention has another hook that works, which is the shape a
        # dropped argument takes in practice.
        @test_logs (
            :warn,
            r"BlockOneGoodOneShort has a method of its own for competing_risk, but at no number of arguments",
        ) match_mode = :any ModelSpec(bp; interventions = [BlockOneGoodOneShort()])
        # A hook written with every argument variadic leaves a `Vararg` where
        # the check looks for the intervention's type. It names no
        # intervention, so the answer is "no method of its own", and neither
        # overload may throw on it.
        @test !EpiBranch._has_own_method(_all_variadic, BlockVariadic, AbstractIntervention)
        @test !EpiBranch._has_own_method(
            _all_variadic, BlockVariadic, AbstractIntervention, 3
        )
        # A child of an abstract intervention whose override drops an
        # argument: the parent's correct method answers the call, so the
        # child's block never applies, and the warning names the child.
        @test_logs (
            :warn,
            r"_FamilyShort has a method of its own for competing_risk, but at no number of arguments the engine calls it with, so that method is never reached: the engine uses _Family's method instead",
        ) match_mode = :any ModelSpec(bp; interventions = [_FamilyShort()])
        # The same family the other way round is fine: the child's own method
        # is the one a four-argument call reaches.
        @test_logs ModelSpec(bp; interventions = [_FamilyBOK()])
        # A convenience overload of a hook for the same type is not a
        # mis-written hook, since the type's own hook answers the call.
        @test_logs ModelSpec(bp; interventions = [_ExtraOverload()])
        # A child of an abstract intervention whose own override forgets
        # the prefix: the parent's method answers the call, so the child's
        # block never applies and the missing prefix has to be named.
        @test_logs (
            :warn,
            r"defines a method of competing_risk for _FamilyUnqualified",
        ) match_mode = :any ModelSpec(bp; interventions = [_FamilyUnqualified()])
        # The same family with the override written correctly is silent.
        @test_logs ModelSpec(bp; interventions = [_FamilyQualified()])
        # A stray unprefixed method alongside a correct one for the same type
        # is a leftover rather than a mistake.
        @test_logs ModelSpec(bp; interventions = [_BothPrefixes()])
        # A hook shared by two types through a `Union` first parameter, with
        # a convenience overload alongside: nothing to report, and the check
        # must not fail trying to name the union.
        @test_logs ModelSpec(bp; interventions = [_UnionShare()])
        # The wrong-arity mistake written once for two types is reported
        # against the union it was written for.
        @test_logs (
            :warn,
            r"Union\{.*_UnionShortB.*\} has a method of its own for competing_risk",
        ) match_mode = :any ModelSpec(bp; interventions = [_UnionShort()])
        # A correct hook written for a union covers each of its members, so an
        # unprefixed leftover for one of them is not the mistake.
        @test_logs ModelSpec(bp; interventions = [_UnionBoth()])
        # With both a parent hook and a wider union hook in reach, the
        # warning names the one dispatch picks.
        @test_logs (
            :warn,
            r"_RankKid has a method of its own for competing_risk.*the engine uses _RankFam's method instead",
        ) match_mode = :any ModelSpec(bp; interventions = [_RankKid()])
        # A parametric family keeps its parameters in the warning, so the
        # dead method and the standing-in one can be told apart.
        @test_logs (
            :warn,
            r"_ParamIv\{Float64\} has a method of its own for competing_risk.*the engine uses _ParamIv's method instead",
        ) match_mode = :any ModelSpec(bp; interventions = [_ParamIv(2.0)])
        # The inert message names the instantiation that reaches nothing,
        # since the family does have a method of its own.
        @test_logs (
            :warn,
            r"_ParamInert\{Int64\} has a method of its own for none of the hooks",
        ) match_mode = :any ModelSpec(bp; interventions = [_ParamInert(2)])
        # The missing-prefix message names the instantiation the stray method
        # was written for, not the composed type.
        @test_logs (
            :warn,
            r"defines a method of competing_risk for _ParamStray\{Float64\}",
        ) match_mode = :any ModelSpec(bp; interventions = [_ParamStray(2.0)])
        # Written `where {D}`, the unprefixed hook is still found, and a
        # convenience overload is still covered by the type's own hook.
        @test_logs (
            :warn,
            r"defines a method of competing_risk for _WhereStray",
        ) match_mode = :any ModelSpec(bp; interventions = [_WhereStray(2.0)])
        @test_logs ModelSpec(bp; interventions = [_WhereOverload(2.0)])
        # The module named is the one the stray method was written in, not
        # the one the intervention type lives in.
        @test_logs (
            :warn,
            r"_StrayHome defines a method of competing_risk for _StrayFam",
        ) match_mode = :any ModelSpec(bp; interventions = [_StrayUser._StrayKid()])
        # `analytic_offspring_effect`'s default throws rather than doing
        # nothing, so the message names the default without describing it.
        @test_logs (
            :warn,
            r"_AnalyticOnly has a method of its own for analytic_offspring_effect.*the package's own default for analytic_offspring_effect instead",
        ) match_mode = :any ModelSpec(bp; interventions = [_AnalyticOnly()])
        # A variadic helper of a hook's name is not a mis-written hook
        # either, and an intervention whose only such method is that helper
        # gets the general message rather than advice that would break it.
        @test_logs ModelSpec(
            bp; interventions = [_VariadicHelpers.HasVariadicHelper(Int[])]
        )
        @test_logs (
            :warn,
            r"OnlyVariadicHelper has a method of its own for none of the hooks",
        ) match_mode = :any ModelSpec(
            bp; interventions = [_VariadicHelpers.OnlyVariadicHelper()]
        )
        # An effect declared for the likelihood alone is an effect, and so is
        # one declared for the closed-form results alone.
        @test_logs ModelSpec(bp; interventions = [_SusceptibilityOnly()])
        @test_logs ModelSpec(bp; interventions = [_HalfThinning()])
        # An intervention acting through a seam the check does not read says
        # so through `has_effect`, which the general warning names.
        @test_logs ModelSpec(bp; interventions = [Scheduled(_OwnSeam(); start_time = 1.0)])
        @test_logs ModelSpec(bp; interventions = [_OwnSeam()])
        @test !EpiBranch.has_effect(_OwnHelpers.NoHooks())
        @test EpiBranch.has_effect(_ActionsHookOnly())
        # `Isolation` implements several hooks properly and stays silent too.
        iso = Isolation(
            onset_to_isolation_delay = Exponential(1.0), duration = Inf
        )
        @test_logs ModelSpec(bp; interventions = [iso])
    end

    @testset "A Reporting step and an observation model can both be set" begin
        # `Reporting` is the one per-case reporting route; a non-per-case
        # observation model, such as `MinimumSize`, composes with it freely.
        bp = BranchingProcess(Poisson(1.5), Exponential(5.0))
        rep = Reporting(delay = Dirac(0.0), probability = 0.5)
        @test ModelSpec(bp; progression = [rep]) isa ModelSpec
        @test ModelSpec(
            bp; progression = [rep], observation = MinimumSize(2)
        ) isa ModelSpec
    end
end
