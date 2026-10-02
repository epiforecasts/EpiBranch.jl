Three CI settings that made a pull request's checks report problems it did not
have. `codecov.yml` now sets a patch target of 80% and a 1% project threshold,
where the defaults asked every diff to beat the project average and failed
`project` on a drop of a hundredth of a point. The docs job takes a repository
wide concurrency group, since every run publishes its preview to the same
`gh-pages` branch and parallel runs raced for that push. And `.gitignore`
covers `*.cov` and `lcov.info`.
