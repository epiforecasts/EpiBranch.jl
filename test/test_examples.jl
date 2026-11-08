# Run every script under examples/ with few simulations, so that a change to
# the public API they call cannot break them unnoticed. The scripts take their
# simulation count from EPIBRANCH_EXAMPLE_NSIM.
@testset "examples" begin
    dir = joinpath(@__DIR__, "..", "examples")
    scripts = filter(f -> endswith(f, ".jl"), readdir(dir))
    @test !isempty(scripts)
    for script in scripts
        @testset "$script" begin
            withenv("EPIBRANCH_EXAMPLE_NSIM" => "10") do
                # Each script runs in its own module so that names do not leak
                # between scripts or into the test suite.
                mod = Module(:ExampleScript)
                @test begin
                    redirect_stdout(devnull) do
                        Base.include(mod, joinpath(dir, script))
                    end
                    true
                end
            end
        end
    end
end
