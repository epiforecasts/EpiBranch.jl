using Documenter
using EpiBranch

makedocs(;
    modules = [EpiBranch],
    sitename = "EpiBranch.jl",
    remotes = nothing,
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
    ),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Getting started" => "tutorials/getting-started.md",
            "Interventions" => "tutorials/interventions.md",
            "Multi-type models" => "tutorials/multi-type.md",
            "Line lists and contacts" => "tutorials/linelist.md",
            "Chain statistics and likelihood" => "tutorials/chains.md",
            "Analytical functions" => "tutorials/analytical.md",
        ],
        "Design" => "design.md",
        "API reference" => "api.md",
    ],
)

deploydocs(;
    repo = "github.com/epiforecasts/EpiBranch.jl.git",
    push_preview = true,
)
