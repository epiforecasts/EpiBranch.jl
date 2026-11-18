using Pkg
const ROOT = joinpath(@__DIR__, "..", "..")
Pkg.develop(
    [
        PackageSpec(path = ROOT),
        PackageSpec(path = joinpath(ROOT, "lib", "EpiNetwork")),
        PackageSpec(path = joinpath(ROOT, "lib", "EpiHouseholds")),
    ]
)
Pkg.instantiate()

using Test
using JET
using Distributions
using EpiBranch
using EpiNetwork
using EpiHouseholds

JET.test_package(EpiBranch; target_modules = (EpiBranch,))

# Run-time dispatch in the Sellke race. A check of all of `simulate` reports
# dispatches that have nothing to do with the race, so the reports are limited
# to those reached through `_open_route!`, which opens each route for a settled
# case with the context the race builds once and reads at every case. An
# untyped entry in that context shows up here as a dispatch on the case's hot
# path.
function open_route_reports(model)
    result = JET.report_opt(simulate, (typeof(model),); target_modules = (EpiBranch,))
    return filter(JET.get_reports(result)) do report
        any(report.vst) do frame
            frame.linfo.def isa Method && frame.linfo.def.name === :_open_route!
        end
    end
end

ring(n) = [[mod1(i - 1, n), mod1(i + 1, n)] for i in 1:n]

@testset "No run-time dispatch in the Sellke race" begin
    models = (
        "branching process" => ModelSpec(
            BranchingProcess(NegBin(2.5, 0.16), LogNormal(1.6, 0.5))
        ),
        "household" => ModelSpec(HouseholdProcess([3, 3], (i, j) -> Exponential(3.0))),
        "network" => ModelSpec(NetworkProcess(ring(10), Exponential(3.0))),
    )
    for (name, model) in models
        @testset "$name" begin
            @test isempty(open_route_reports(model))
        end
    end
end
