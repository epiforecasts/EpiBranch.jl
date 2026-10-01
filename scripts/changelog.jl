#!/usr/bin/env julia
#
# Assemble the changelog fragments in `changelog.d/` (see its README).
#
#   julia scripts/changelog.jl                 # print the pending section
#   julia scripts/changelog.jl --list          # name the fragments it reads
#   julia scripts/changelog.jl 0.2.0           # fold them into CHANGELOG.md
#
# A release inserts its section at a literal marker and reads nothing else in
# the file. An earlier version parsed the existing `## [Unreleased]` section so
# it could merge entries into it, which meant inferring where headings and
# sections began from the shape of a line: a `## ` or `### ` line inside a
# fenced code block was taken for a heading, and a heading the pattern did not
# match filed the next entry under the wrong one. A marker cannot misread the
# document.

using Dates

const CATEGORIES = ["added", "changed", "deprecated", "removed", "fixed", "security"]

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FRAGMENT_DIR = joinpath(ROOT, "changelog.d")
const CHANGELOG = joinpath(ROOT, "CHANGELOG.md")
const MARKER = "<!-- releases go below this line -->"

struct Fragment
    category::String
    name::String
    text::String
end

# An editor backup, an auto-save or a merge leftover is nobody's entry, and
# `.gitignore` does not cover them, so they are passed over. `git mergetool`
# writes its copies as `added-foo.BACKUP.4321.md`, which would otherwise parse
# as a fragment and ship as a duplicate bullet.
const STRAY = r"(^#.*#$)|(~$)|\.(orig|rej|bak|swp|swo)$|\.(BACKUP|BASE|LOCAL|REMOTE)\.\d+\."

_ignored(name) = name == "README.md" || startswith(name, ".") || occursin(STRAY, name)

# Every other name in the directory is meant as a fragment, so one this cannot
# use is an error. A file quietly skipped is an entry its author believes is
# recorded, and nothing would say otherwise until the release.
function read_fragments(dir)
    fragments = Fragment[]
    for name in sort(readdir(dir))
        _ignored(name) && continue
        path = joinpath(dir, name)
        isdir(path) && error(
            "changelog.d/$name is a directory. Fragments sit directly in changelog.d/."
        )
        endswith(name, ".md") && occursin('-', name) || error(
            "changelog.d/$name is not named <category>-<slug>.md."
        )
        category = first(split(name, '-'; limit = 2))
        category in CATEGORIES || error(
            "changelog.d/$name: '$category' is not one of $(join(CATEGORIES, ", ")). " *
                "Name the file <category>-<slug>.md."
        )
        text = strip(read(path, String))
        isempty(text) && error("changelog.d/$name is empty.")
        push!(fragments, Fragment(category, name, text))
    end
    return fragments
end

# A bullet whose continuation lines are indented to sit under it, so the
# rendered list survives a fragment that spans paragraphs.
function as_bullet(text)
    lines = split(text, '\n')
    io = IOBuffer()
    println(io, "- ", lines[1])
    for line in lines[2:end]
        println(io, isempty(strip(line)) ? "" : "  " * line)
    end
    return String(take!(io))
end

function render(fragments)
    io = IOBuffer()
    for category in CATEGORIES
        selected = filter(f -> f.category == category, fragments)
        isempty(selected) && continue
        println(io, "### ", uppercasefirst(category), "\n")
        for fragment in selected
            print(io, as_bullet(fragment.text))
            println(io)
        end
    end
    return rstrip(String(take!(io)))
end

function release!(version, fragments)
    # A release deletes every fragment, so the argument is checked before
    # anything is unlinked.
    occursin(r"^v?\d", version) ||
        error("'$version' does not look like a version, e.g. `changelog-release -- 0.2.0`.")
    isempty(fragments) && error("no fragments in changelog.d/: nothing to release.")
    source = read(CHANGELOG, String)
    occursin("## [$version]", source) &&
        error("$CHANGELOG already has a '## [$version]' section.")
    marker = findfirst(MARKER, source)
    marker === nothing && error("$CHANGELOG has no '$MARKER' line.")

    # Whatever a maintainer wrote above the marker stays there. Warned about
    # rather than moved, because moving it would mean reading the document's
    # structure, which is what this script no longer does.
    above = source[1:prevind(source, first(marker))]
    unreleased = findlast("## [Unreleased]", above)
    if unreleased !== nothing && occursin(r"^\s*[-*] "m, above[last(unreleased):end])
        @warn "Entries sit under `## [Unreleased]` above the marker. They stay " *
            "there; move them under `## [$version]` yourself if they belong to it."
    end

    heading = "## [$version] - $(Dates.format(Dates.today(), "yyyy-mm-dd"))"
    section = string("\n\n", heading, "\n\n", render(fragments))
    write(
        CHANGELOG,
        string(source[1:last(marker)], section, source[(last(marker) + 1):end])
    )
    for fragment in fragments
        rm(joinpath(FRAGMENT_DIR, fragment.name))
    end
    println("Released ", length(fragments), " fragment(s) under $version.")
    return nothing
end

function main(args)
    isdir(FRAGMENT_DIR) || error("no changelog.d/ directory at $FRAGMENT_DIR.")
    if !isempty(args) && startswith(first(args), "-")
        args == ["--list"] ||
            error("unknown option '$(first(args))'. Pass --list, or a version.")
        foreach(f -> println(f.name), read_fragments(FRAGMENT_DIR))
        return 0
    end
    fragments = read_fragments(FRAGMENT_DIR)
    if isempty(args)
        isempty(fragments) ? println("No changelog fragments pending.") :
            println(render(fragments))
        return 0
    end
    release!(only(args), fragments)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main(ARGS))
end
