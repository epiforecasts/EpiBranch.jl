# Installation

The package is in the General registry:

```julia
using Pkg
Pkg.add("EpiBranch")
```

The current household and network packages require EpiBranch 0.2. The registered
0.1.0 release lacks interfaces they use. Until 0.2 is registered, install all
three packages from the same revision in a fresh project (Julia 1.11 or later):

```julia
using Pkg
Pkg.activate("epibranch-example"; shared=false)
repo = "https://github.com/epiforecasts/EpiBranch.jl"
rev = "9e1c3be3505202fce50679ee43cd10efd66a8690"  # matching development versions before the 0.2 release
Pkg.add([
    PackageSpec(url=repo, rev=rev),
    PackageSpec(url=repo, rev=rev, subdir="lib/EpiHouseholds"),
    PackageSpec(url=repo, rev=rev, subdir="lib/EpiNetwork"),
])
```

Keep the generated `Project.toml` and `Manifest.toml` with an analysis to preserve
the resolved revisions. When choosing a newer development commit, use its same
full revision for every package.

After EpiBranch 0.2.0 is registered and the matching `v0.2.0` tag is published,
the release installation will be:

```julia
Pkg.add([
    PackageSpec(name="EpiBranch", version="0.2.0"),
    PackageSpec(url=repo, rev="v0.2.0", subdir="lib/EpiHouseholds"),
    PackageSpec(url=repo, rev="v0.2.0", subdir="lib/EpiNetwork"),
])
```

The [installation checks](https://github.com/epiforecasts/EpiBranch.jl/blob/main/scripts/check_installation.jl) test both companion
packages in a temporary project. CI checks matching source revisions and rejects
the incompatible registry version. Maintainers can run the release installation
workflow after registration; it resolves EpiBranch from General without using a
local checkout.

