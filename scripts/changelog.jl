#!/usr/bin/env julia
#
# Assemble the changelog fragments in `changelog.d/` (see its README).
#
#   julia scripts/changelog.jl                 # print the pending section
#   julia scripts/changelog.jl 0.2.0           # fold them into CHANGELOG.md

using Dates

const CATEGORIES = ["added", "changed", "deprecated", "removed", "fixed", "security"]

const ROOT = normpath(joinpath(@__DIR__, ".."))
const FRAGMENT_DIR = joinpath(ROOT, "changelog.d")
const CHANGELOG = joinpath(ROOT, "CHANGELOG.md")
const UNRELEASED = "## [Unreleased]"

struct Fragment
    category::String
    name::String
    text::String
end

function read_fragments(dir)
    fragments = Fragment[]
    for name in sort(readdir(dir))
        (endswith(name, ".md") && name != "README.md") || continue
        category = first(split(name, '-'; limit = 2))
        category in CATEGORIES || error(
            "changelog.d/$name: '$category' is not one of $(join(CATEGORIES, ", ")). " *
                "Name the file <category>-<slug>.md."
        )
        text = strip(read(joinpath(dir, name), String))
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

# Split what sits under `## [Unreleased]` into the prose before its first `###`
# heading, which describes the unreleased state and stays put, and the entries
# under each heading, which belong to whichever version is cut next.
function split_unreleased(body)
    preamble = IOBuffer()
    entries = Dict{String, String}()
    order = String[]
    current = ""
    buffer = IOBuffer()
    function flush_current()
        isempty(current) && return nothing
        text = rstrip(String(take!(buffer)))
        if !isempty(text)
            # Appended, never assigned: a merge leaves two `### Added` blocks
            # under one heading, and assigning would drop the first.
            entries[current] = haskey(entries, current) ?
                entries[current] * "\n" * text : text
            current in order || push!(order, current)
        end
        return nothing
    end
    for line in split(body, '\n')
        heading = match(r"^###\s+([\w ]+?)\s*$", line)
        if heading !== nothing
            flush_current()
            current = lowercase(heading.captures[1])
            continue
        end
        println(isempty(current) ? preamble : buffer, line)
    end
    flush_current()
    return rstrip(String(take!(preamble))), entries, order
end

function render(fragments, existing = Dict{String, String}(), order = String[])
    io = IOBuffer()
    # The six standard categories first, then any other heading the file already
    # had, in the order it had them. Carried through rather than dropped: this
    # rewrites the project's history, so an unrecognised heading must survive.
    extra = filter(c -> !(c in CATEGORIES), order)
    for category in vcat(CATEGORIES, extra)
        selected = filter(f -> f.category == category, fragments)
        carried = get(existing, category, "")
        (isempty(selected) && isempty(carried)) && continue
        println(io, "### ", uppercasefirst(category), "\n")
        isempty(carried) || println(io, carried, "\n")
        for fragment in selected
            print(io, as_bullet(fragment.text))
            println(io)
        end
    end
    return rstrip(String(take!(io)))
end

function release!(version, fragments)
    isempty(strip(version)) && error("give a version, e.g. `changelog-release -- 0.2.0`.")
    source = read(CHANGELOG, String)
    occursin("## [$version]", source) &&
        error("$CHANGELOG already has a '## [$version]' section.")
    marker = findfirst(UNRELEASED, source)
    marker === nothing && error("$CHANGELOG has no '$UNRELEASED' heading.")
    rest = source[(last(marker) + 1):end]
    next_section = findfirst("\n## ", rest)
    # `prevind`, because a line ending in a multibyte character before a `##`
    # heading would make a byte index land mid-character and throw.
    body = next_section === nothing ? rest : rest[1:prevind(rest, first(next_section))]
    tail = next_section === nothing ? "" : rest[first(next_section):end]

    preamble, existing, order = split_unreleased(body)
    entries = render(fragments, existing, order)
    isempty(entries) &&
        error("nothing to release: no fragments, and no entries under $UNRELEASED.")

    heading = "## [$version] - $(Dates.format(Dates.today(), "yyyy-mm-dd"))"
    rebuilt = string(
        source[1:last(marker)], "\n\n",
        isempty(preamble) ? "" : preamble * "\n\n",
        heading, "\n\n", entries, "\n", tail
    )
    # Sections that were empty on one side or the other leave runs of blank
    # lines behind; a changelog never wants more than one.
    write(CHANGELOG, replace(rebuilt, r"\n{3,}" => "\n\n"))
    for fragment in fragments
        rm(joinpath(FRAGMENT_DIR, fragment.name))
    end
    println(
        "Released ", length(fragments), " fragment(s) under $version",
        isempty(existing) ? "" : ", merged with the entries already under Unreleased", "."
    )
    return nothing
end

function main(args)
    isdir(FRAGMENT_DIR) || error("no changelog.d/ directory at $FRAGMENT_DIR.")
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
