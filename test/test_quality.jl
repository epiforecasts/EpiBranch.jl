using Aqua
using Documenter
using Pkg
using Runic

@testset "Code quality (Aqua.jl)" begin
    Aqua.test_all(
        EpiBranch;
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
    root = joinpath(@__DIR__, "..")
    unformatted = String[]
    checked = 0
    for (dir, _, files) in walkdir(root)
        # The rendered site is generated, and `.git` holds no source. The match
        # is on a path component, since a checkout's own path can contain
        # `.git`.
        parts = splitpath(relpath(dir, root))
        (occursin(joinpath("docs", "build"), dir) || ".git" in parts) && continue
        for file in files
            endswith(file, ".jl") || continue
            path = joinpath(dir, file)
            source = read(path, String)
            checked += 1
            # `filemode = true` is what `Runic.main` uses, so the hook and this
            # gate agree about a file's leading and trailing newlines.
            Runic.format_string(source; filemode = true) == source ||
                push!(unformatted, relpath(path, root))
        end
    end
    @test unformatted == String[]
    # A walk that matched nothing would otherwise pass silently.
    @test checked > 100
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
