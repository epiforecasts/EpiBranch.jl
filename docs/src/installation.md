# Installation

## Before you start

Install Julia with [juliaup](https://julialang.org/install/), the official
installer. EpiBranch needs Julia 1.10 or later, and the household and network
packages need Julia 1.11 or later. Start Julia by typing `julia` in a terminal,
or use an editor such as VS Code with the Julia extension.

Julia keeps the packages for each analysis in a project folder, much as
[renv](https://rstudio.github.io/renv/) does for R. The project records which
package versions you used in two files, `Project.toml` and `Manifest.toml`.
Keep them with your analysis so it can be rerun later with the same versions.

The first time you load EpiBranch, and the first time you call a function in a
new session, Julia compiles code, which can take from a few seconds to a minute.
Later calls in the same session run at full speed.

You do not need R to use EpiBranch. [Julia for R users](julia-for-r-users.md)
explains the syntax used in these pages.

## The core package only

EpiBranch is in the General registry, Julia's equivalent of CRAN:

```julia
using Pkg
Pkg.activate("my-analysis")  # use (or create) the project folder my-analysis
Pkg.add("EpiBranch")
```

Then load it with `using EpiBranch`, as you would `library()` a package in R.

## With the household and network packages

The household and network packages (EpiHouseholds and EpiNetwork) need a newer
version of EpiBranch (0.2) than the one registered (0.1.0). Until 0.2 is
released, install all three from GitHub at the same commit:

```julia
using Pkg
Pkg.activate("my-analysis")
repo = "https://github.com/epiforecasts/EpiBranch.jl"
# A commit on which the three packages work together
rev = "9e1c3be3505202fce50679ee43cd10efd66a8690"
Pkg.add([
    PackageSpec(url = repo, rev = rev),
    PackageSpec(url = repo, rev = rev, subdir = "lib/EpiHouseholds"),
    PackageSpec(url = repo, rev = rev, subdir = "lib/EpiNetwork"),
])
```

`PackageSpec` says where to install a package from: here the GitHub repository,
a commit (`rev`), and for the two companion packages the folder within the
repository that holds them (`subdir`). To use a newer version, replace `rev`
with the full identifier (hash) of a later commit from the
[commit list on GitHub](https://github.com/epiforecasts/EpiBranch.jl/commits/main),
and use the same one for all three packages.

Once EpiBranch 0.2 is released this becomes an ordinary `Pkg.add`.
