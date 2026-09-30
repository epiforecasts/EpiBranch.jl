# Changelog fragments

A pull request records its changelog entry as a file in this directory rather
than by editing `CHANGELOG.md`. Two pull requests then never touch the same
lines, so neither conflicts with the other, and a merge cannot resurrect an
entry a release removed.

## Adding an entry

Create `changelog.d/<category>-<slug>.md`, where `<category>` is one of the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) headings in lower case:

    added  changed  deprecated  removed  fixed  security

The slug is yours; anything that will not collide with another open pull
request does. The file holds the entry's text, without the leading `- `:

```markdown
`Isolation`'s `onset_to_isolation_delay` accepts a `Real`, a `Distribution`,
or a function `(rng, ind) -> Real`.
```

Several paragraphs and nested bullets are fine. The text is indented under its
bullet when assembled, so write it flush left.

## Reading them

    task changelog          # what the next release's section will say
    task changelog-release -- 0.2.0

`changelog-release` folds every fragment into `CHANGELOG.md` under a new version
heading and deletes the fragments. Entries already sitting under
`## [Unreleased]` are merged in by category, so each heading appears once. The
prose above those headings describes the unreleased state, so it stays where it
is; edit it yourself if the release makes it wrong. Run this on the release
branch and read the result before committing.
