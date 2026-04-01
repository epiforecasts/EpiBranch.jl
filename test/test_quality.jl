using Aqua
using ExplicitImports
using Documenter

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(EpiBranch;
        ambiguities=false,
        piracies=false,
        deps_compat=(ignore=[:Dates, :Random],),
    )
end

@testset "Explicit imports" begin
    @test check_no_stale_explicit_imports(EpiBranch) === nothing
end

@testset "Docstring examples" begin
    doctest(EpiBranch; manual=false)
end
