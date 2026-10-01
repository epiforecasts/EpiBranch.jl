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

Everything in this directory is read as an entry, so a file that cannot be read
fails CI rather than being passed over: a name without one of the categories
above, a subdirectory, an empty file. Ignored are names beginning with a dot,
and the leavings of an editor or a merge — `#...#`, `...~`, `.bak`, `.swp`,
`.swo`, `.orig`, `.rej`, and `git mergetool`'s `.BACKUP.1234.` copies.

## Carrying no entry

CI fails a pull request that adds no fragment. A change that carries none — one
touching only CI or tests, and the release pull request, which removes the
fragments — takes the `no changelog` label. The label excuses the entry alone;
the fragments the pull request touched are still read.

## Reading them

    task changelog          # what the next release's section will say
    task changelog-release -- 0.2.0

`changelog-release` writes a new version section at the
`<!-- releases go below this line -->` marker in `CHANGELOG.md` and deletes the
fragments. The marker is the only thing it looks for, so where a line sits
relative to it decides that line's fate:

- **Below the marker**, and so inside the new release: the fragments, and any
  entry written by hand under `## [Unreleased]`. That is how the entries
  already in `CHANGELOG.md` will ship with 0.2.0.
- **Above the marker**, and so left under `## [Unreleased]`: the prose
  describing the unreleased state. An entry written up there stays behind after
  the release, which is rarely meant, so the script warns about one.

Run this on the release branch and read the result before committing. A release
that carries hand-written entries gets two `### Added` headings, one from the
fragments and one from those entries; merge them yourself.
