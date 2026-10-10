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

EpiBranch is in the General registry, Julia's equivalent of CRAN, and you load
it with `using EpiBranch`, as you would `library()` a package in R.

!!! warning "The registered version is older than these pages"
    The registered version (0.1.0) does not run the examples in these pages,
    which describe version 0.2: setting how long isolation or quarantine lasts
    (`duration`), for example, fails with an error. Until 0.2 is released,
    install EpiBranch from GitHub as in the next section, leaving out the
    household and network packages if you do not need them.

## With the household and network packages

The household and network packages (EpiHouseholds and EpiNetwork) need a newer
version of EpiBranch (0.2) than the one registered (0.1.0). Until 0.2 is
released, install all three from the development version on GitHub (the
`main` branch):

```julia
using Pkg
Pkg.activate("my-analysis")
repo = "https://github.com/epiforecasts/EpiBranch.jl"
# The development version, which these pages describe
rev = "main"
Pkg.add([
    PackageSpec(url = repo, rev = rev),
    PackageSpec(url = repo, rev = rev, subdir = "lib/EpiHouseholds"),
    PackageSpec(url = repo, rev = rev, subdir = "lib/EpiNetwork"),
])
```

`PackageSpec` says where to install a package from: here the GitHub repository,
a branch or commit (`rev`), and for the two companion packages the folder within
the repository that holds them (`subdir`). Running `Pkg.update()` later brings
all three up to the latest development version. To fix an analysis to one
version instead, set `rev` to the full identifier (hash) of a commit from the
[commit list on GitHub](https://github.com/epiforecasts/EpiBranch.jl/commits/main),
the same one for all three packages.

Once EpiBranch 0.2 is released this becomes an ordinary `Pkg.add`.
