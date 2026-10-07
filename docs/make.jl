using Documenter
using DocumenterVitepress
using EpiBranch
using EpiNetwork
using EpiHouseholds

makedocs(;
    modules = [EpiBranch, EpiNetwork, EpiHouseholds],
    sitename = "EpiBranch.jl",
    authors = "epiforecasts contributors",
    remotes = nothing,
    warnonly = [:missing_docs, :docs_block],
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/epiforecasts/EpiBranch.jl",
        devbranch = "main",
        devurl = "dev"
    ),
    pages = [
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Julia for R users" => "julia-for-r-users.md",
        "Tutorials" => [
            "Getting started" => "tutorials/getting-started.md",
            "Interventions" => [
                "Isolation and contact tracing" => "tutorials/interventions.md",
                "Vaccination" => "tutorials/vaccination.md",
            ],
            "Clinical transitions" => "tutorials/transitions.md",
            "Transmission models" => [
                "Multi-type models" => "tutorials/multi-type.md",
                "Network models" => "tutorials/networks.md",
                "Household models" => "tutorials/households.md",
                "Homogeneous models" => "tutorials/homogeneous.md",
                "Covariates and time-varying transmission" => "tutorials/covariate-transmission.md",
            ],
            "Line lists and contacts" => "tutorials/linelist.md",
            "Analysis" => [
                "Chain statistics" => "tutorials/chains.md",
                "Analytical functions" => "tutorials/analytical.md",
                "Inference" => "tutorials/inference.md",
            ],
            "Extending EpiBranch" => "tutorials/extending.md",
        ],
        "Design" => "design.md",
        "Glossary" => "glossary.md",
        "API reference" => "api.md",
    ]
)

# A pull request's preview goes to the same `gh-pages` branch that every other
# open pull request's docs job writes to, and the preview cleanup workflow
# force-pushes there as well, so overlapping runs lose a push. Each attempt
# re-fetches the branch in a fresh temporary clone, so retrying lands the
# preview against whatever is there by then. Only a rejected `git push` is
# retried, and after two rejections the run warns and succeeds, because the
# build is what this job reports on and the published documentation is
# unaffected by a missing preview. Any other failure ends the job. On `main` the
# deploy is the published documentation, so it gets one attempt and any failure
# ends the job.
function deploy()
    return DocumenterVitepress.deploydocs(;
        repo = "github.com/epiforecasts/EpiBranch.jl",
        devbranch = "main",
        push_preview = true
    )
end

"""Whether `e` is a `git push` that the remote rejected."""
push_rejected(e) = e isa ProcessFailedException &&
    any(p -> "push" in p.cmd.exec, e.procs)

if get(ENV, "EPIBRANCH_DOCS_PREVIEW_BEST_EFFORT", "false") == "true"
    for attempt in 1:2
        try
            deploy()
            break
        catch e
            push_rejected(e) || rethrow()
            if attempt == 2
                println(
                    "::warning title=Documentation preview::The preview push " *
                        "to gh-pages was rejected twice; the build succeeded."
                )
                @warn "The documentation preview push was rejected twice. The " *
                    "build succeeded; something else writing to `gh-pages` " *
                    "most likely won both pushes." exception = (e, catch_backtrace())
            else
                @info "The documentation preview push was rejected; retrying."
            end
        end
    end
else
    deploy()
end
