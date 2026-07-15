# End-to-end check that the timeline (transitions), the infectiousness
# windows, the from-gate, and the until-censor compose into funeral
# transmission: a second infectious window between death and burial that
# only the dead open and that safe burial would shorten.
@testset "Funeral transmission end-to-end" begin
    history = [
        Transition(:infectious, from = :infection, delay = LogNormal(1.0, 0.3)),
        Transition(:onset, from = :infection, delay = LogNormal(1.6, 0.4)),
        Transition(:died, from = :onset, delay = Gamma(2.0, 3.0),
            probability = 0.6, terminal = true),
        Transition(:recovered, from = :onset, delay = Gamma(2.0, 5.0), terminal = true),
        Transition(:buried, from = :died, delay = Gamma(2.0, 1.0))
    ]
    community = Infectiousness(NegBin(1.5, 0.5);
        from = :infectious, until = (:recovered, :died), kernel = Gamma(2.0, 2.0))
    funeral = Infectiousness(NegBin(1.0, 0.5);
        from = :died, until = (:buried,), kernel = Gamma(2.0, 0.5))
    opts = (; n_initial = 30, max_cases = 3000)

    s = simulate(ModelSpec(BranchingProcess(community, funeral); progression = history);
        opts..., rng = StableRNG(1))

    # An infected case transmitted at the funeral if its infector had died
    # and it was infected at or after that death.
    funeral_kids = filter(s.individuals) do ind
        (ind.parent_id == 0 || !is_infected(ind)) && return false
        p = s.individuals[ind.parent_id]
        get(p.state, :died, false) && ind.infection_time >= get(p.state, :died_time, Inf)
    end
    @test !isempty(funeral_kids)

    for ind in funeral_kids
        p = s.individuals[ind.parent_id]
        @test p.state[:died] == true                          # from-gate: only the dead
        @test ind.infection_time < p.state[:buried_time]      # censor: before burial
    end

    # No infected contact of a recovered (never-dead) case is a funeral
    # contact: the funeral window never opened for them.
    for ind in s.individuals
        (ind.parent_id == 0 || !is_infected(ind)) && continue
        p = s.individuals[ind.parent_id]
        if get(p.state, :recovered, false) && !get(p.state, :died, false)
            @test ind.infection_time < get(p.state, :recovered_time, Inf)
        end
    end

    # The funeral window adds transmission: at least as many cases as the
    # community-only model on the same seed.
    s_no = simulate(ModelSpec(BranchingProcess(community); progression = history);
        opts..., rng = StableRNG(1))
    @test s.cumulative_cases >= s_no.cumulative_cases
end
