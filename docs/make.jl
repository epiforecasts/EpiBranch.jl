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
        "Tutorials" => [
            "Getting started" => "tutorials/getting-started.md",
            "Interventions" => "tutorials/interventions.md",
            "Clinical transitions" => "tutorials/transitions.md",
            "Multi-type models" => "tutorials/multi-type.md",
            "Network models" => "tutorials/networks.md",
            "Contextual pair kernels" => "tutorials/pair_kernels.md",
            "Household models" => "tutorials/households.md",
            "Homogeneous models" => "tutorials/homogeneous.md",
            "Line lists and contacts" => "tutorials/linelist.md",
            "Chain statistics" => "tutorials/chains.md",
            "Inference" => "tutorials/inference.md",
            "Analytical functions" => "tutorials/analytical.md",
            "Extending EpiBranch" => "tutorials/extending.md",
        ],
        "Design" => "design.md",
        "API reference" => "api.md",
    ]
)

# A pull request's preview goes to the same `gh-pages` branch that every other
# open pull request's docs job writes to, and the preview cleanup workflow
# force-pushes there as well, so overlapping runs lose a push. Each attempt
# re-fetches the branch in a fresh temporary clone, so retrying lands the
# preview against whatever is there by then. Only after both attempts does the
# run give up and warn, because the build is what this job reports on and the
# published documentation is unaffected by a missing preview. On `main` the
# deploy is the published documentation, so it gets one attempt and any failure
# ends the job.
function deploy()
    return DocumenterVitepress.deploydocs(;
        repo = "github.com/epiforecasts/EpiBranch.jl",
        devbranch = "main",
        push_preview = true
    )
end

if get(ENV, "EPIBRANCH_DOCS_PREVIEW_BEST_EFFORT", "false") == "true"
    for attempt in 1:2
        try
            deploy()
            break
        catch e
            if attempt == 2
                @warn "Publishing the documentation preview failed twice. The " *
                    "build succeeded; something else writing to `gh-pages` " *
                    "most likely won both pushes." exception = (e, catch_backtrace())
            else
                @info "Publishing the documentation preview failed; retrying."
            end
        end
    end
else
    deploy()
end
