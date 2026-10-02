Changelog entries live in `changelog.d/` as one file per pull request, which
`scripts/changelog.jl` folds into this file at release. Two pull requests no
longer touch the same lines, so neither conflicts with the other.
