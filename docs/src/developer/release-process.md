# Release process

EpiBranch and its two satellite packages, EpiNetwork and EpiHouseholds under
`lib/`, are released from `main` through the General registry. EpiBranch goes
first, because the satellites depend on it.

## Releasing EpiBranch

1. Open a release pull request that sets `version` in `Project.toml` and folds
   the changelog fragments into `CHANGELOG.md` with
   `task changelog-release -- X.Y.Z`. This deletes the fragments; apply the
   `no changelog` label to the pull request.
2. Merge it. Comment `/register` on any issue or pull request, or run the
   Register workflow from the Actions tab. The workflow posts
   `@JuliaRegistrator register` on the head of `main`. JuliaRegistrator opens
   the registry pull request. Once it merges, TagBot creates the GitHub
   release.

## Releasing a satellite

Each satellite has its own `Project.toml`, with its own version and a compat
bound on EpiBranch; bump those in the release pull request. Once the EpiBranch
version it needs is in the registry, comment `/register lib/EpiNetwork` or
`/register lib/EpiHouseholds`, and the workflow adds the `subdir=` the
Registrator needs. Register each satellite separately. TagBot has a step per
satellite and tags them as `EpiNetwork-vX.Y.Z` and `EpiHouseholds-vX.Y.Z`
once their registry pull requests merge.

## Release notes

`CHANGELOG.md` holds the folded entries. The GitHub release that TagBot creates
has a diff link to the previous tag, the merged pull requests and closed issues
since it, and any release notes written under the `@JuliaRegistrator register`
comment.
