using Documenter
using DocumenterVitepress
using EpiBranch
using EpiNetwork

makedocs(;
    modules = [EpiBranch, EpiNetwork],
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
            "Network models" => "tutorials/networks.md",
            "Line lists and contacts" => "tutorials/linelist.md",
            "Chain statistics" => "tutorials/chains.md",
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
