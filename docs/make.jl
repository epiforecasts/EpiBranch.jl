using Documenter
using DocumenterVitepress
using EpiBranch
using EpiBranchCore
using EpiBranchProcess
using EpiInterventions
using EpiTransitions
using EpiObservation
using EpiOutput
using EpiAnalytics

makedocs(;
    modules = [
        EpiBranch,
        EpiBranchCore,
        EpiBranchProcess,
        EpiInterventions,
        EpiTransitions,
        EpiObservation,
        EpiOutput,
        EpiAnalytics
    ],
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
        "Tutorials" => [
            "Getting started" => "tutorials/getting-started.md",
            "Interventions" => "tutorials/interventions.md",
            "Clinical transitions" => "tutorials/transitions.md",
            "Multi-type models" => "tutorials/multi-type.md",
            "Line lists and contacts" => "tutorials/linelist.md",
            "Chain statistics, likelihood, and fitting" => "tutorials/chains.md",
            "Inference" => "tutorials/inference.md",
            "Analytical functions" => "tutorials/analytical.md",
            "Extending EpiBranch" => "tutorials/extending.md"
        ],
        "Design principles" => "principles.md",
        "Design" => "design.md",
        "Benchmarks" => "benchmarks.md",
        "API reference" => "api.md"
    ]
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/epiforecasts/EpiBranch.jl",
    devbranch = "main",
    push_preview = true
)
