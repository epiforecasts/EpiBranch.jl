using Aqua
using Documenter
using JuliaFormatter
using Pkg

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(EpiBranch;
        ambiguities = false,
        piracies = false,
        deps_compat = (ignore = [:Dates, :Random],)
    )
end

@testset "Explicit imports" begin
    if VERSION >= v"1.11"
        using ExplicitImports
        @test check_no_stale_explicit_imports(EpiBranch) === nothing
    else
        @info "Skipping ExplicitImports on Julia $VERSION"
        @test_skip true
    end
end

@testset "Docstring examples" begin
    doctest(EpiBranch; manual = false)
end

@testset "Code formatting" begin
    @test format(joinpath(@__DIR__, ".."); overwrite = false) == true
end

@testset "Code linting (JET)" begin
    jet_env = joinpath(@__DIR__, "jet")
    cmd = `$(Base.julia_cmd()) --project=$jet_env $(joinpath(jet_env, "runtests.jl"))`
    exitcode = try
        run(pipeline(cmd; stdout = stdout, stderr = stderr))
        0
    catch
        1
    end
    @test exitcode == 0
end
