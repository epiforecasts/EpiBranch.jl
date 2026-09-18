# Each invocation starts in a temporary project. The release mode resolves the
# root package from General, without developing a local checkout.
using Pkg
using TOML
using Test

mode = get(ARGS, 1, "source")
mode in ("source", "released", "reject-old") || error("unknown installation mode: $mode")
repo = get(ARGS, 2, "https://github.com/epiforecasts/EpiBranch.jl")
revision = get(ARGS, 3, "v0.2.0")
root_version = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
Pkg.activate(; temp = true)

companions = [PackageSpec(; url = repo, rev = revision, subdir = "lib/$name")
              for name in ("EpiHouseholds", "EpiNetwork")]
if mode == "reject-old"
    # Only a dependency-resolution error satisfies this check; network or load
    # errors should fail CI rather than masquerading as compatibility checks.
    @test_throws Pkg.Resolve.ResolverError Pkg.add([PackageSpec(; name = "EpiBranch", version = v"0.1.0");
                                                    companions])
else
    root = mode == "released" ?
           PackageSpec(; name = "EpiBranch", version = root_version) :
           PackageSpec(; url = repo, rev = revision)
    Pkg.add([root; companions; PackageSpec(; name = "Distributions")])
    dependencies = Pkg.dependencies()
    root_info = only(info for info in values(dependencies) if info.name == "EpiBranch")
    @test !root_info.is_tracking_path
    @test mode != "released" || !root_info.is_tracking_repo
    @eval using EpiBranch, EpiNetwork, EpiHouseholds, Distributions, Random
    network = NetworkProcess([Int[], Int[]], Exponential(2.0))
    @test simulate(network; rng = Xoshiro(42)).cumulative_cases == 1
    household = ModelSpec(HouseholdProcess([2], Exponential(2.0));
        progression = [Transition(:recovered; from = :infection, delay = 0.0,
            terminal = true)])
    @test simulate(household; rng = Xoshiro(42)).cumulative_cases == 1
end
