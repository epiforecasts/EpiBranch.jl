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

# A pull request's preview is published to the same `gh-pages` branch every
# other open pull request's docs job pushes to, so two overlapping runs race and
# one loses the push. The build is what this job gates on, and a lost preview
# says nothing about the documentation, so on a pull request the deploy is
# best-effort and a failure is reported as a warning. A deploy of the released
# or development documentation still fails the job.
function deploy()
    return DocumenterVitepress.deploydocs(;
        repo = "github.com/epiforecasts/EpiBranch.jl",
        devbranch = "main",
        push_preview = true
    )
end

if get(ENV, "EPIBRANCH_DOCS_PREVIEW_BEST_EFFORT", "false") == "true"
    try
        deploy()
    catch e
        @warn "Publishing the documentation preview failed; the build itself " *
            "succeeded. Another pull request's docs job most likely pushed to " *
            "`gh-pages` first." exception = (e, catch_backtrace())
    end
else
    deploy()
end
