# Compare the R functions and arguments mapped in docs/src/from-r-packages.md
# with the latest release of each package on GitHub.
#
# Usage: julia docs/check_r_mapping.jl [package ...]
#
# Reads docs/r_mapping.toml, downloads the source of each package's latest
# release, and reports:
#   - a release newer than the one the mapping was checked against,
#   - mapped functions that the release no longer exports,
#   - mapped arguments that the function no longer has,
#   - exports and arguments of mapped functions that the list does not have.
# The exit status is 1 if anything differs. Set GITHUB_TOKEN to raise the API
# rate limit. This script is not part of the docs build or CI: run it before
# each release (see docs/src/contributing.md).

using Downloads
using TOML

const MAPPING = joinpath(@__DIR__, "r_mapping.toml")

function github_get(url::AbstractString, output)
    headers = ["Accept" => "application/vnd.github+json"]
    token = get(ENV, "GITHUB_TOKEN", "")
    isempty(token) || push!(headers, "Authorization" => "Bearer $token")
    return Downloads.download(url, output; headers = headers)
end

"""Tag of the latest release of `repo` (`owner/name`)."""
function latest_tag(repo)
    file = github_get("https://api.github.com/repos/$repo/releases/latest", tempname())
    m = match(r"\"tag_name\"\s*:\s*\"([^\"]+)\"", read(file, String))
    m === nothing && error("no release found for $repo")
    return String(m.captures[1])
end

"""Download and unpack the source of `repo` at `tag`; return its directory."""
function fetch_source(repo, tag)
    archive = github_get("https://api.github.com/repos/$repo/tarball/$tag", tempname())
    dir = mktempdir()
    run(`tar -xzf $archive -C $dir --strip-components=1`)
    return dir
end

"""Names exported in the package's NAMESPACE file."""
function exports(dir)
    text = read(joinpath(dir, "NAMESPACE"), String)
    names = String[]
    for m in eachmatch(r"^export\(([^)]*)\)"m, text)
        push!(names, strip(m.captures[1], ['"', '`', ' ']))
    end
    return sort(unique(names))
end

"""The text inside the braces that open at `start` (the index of `{`), and the
index after the closing brace."""
function braced(text, start)
    depth = 0
    i = start
    while i <= lastindex(text)
        c = text[i]
        if c == '\\'
            i = nextind(text, i)
        elseif c == '{'
            depth += 1
        elseif c == '}'
            depth -= 1
            depth == 0 && return text[nextind(text, start):prevind(text, i)], nextind(text, i)
        end
        i = nextind(text, i)
    end
    error("unbalanced braces in an Rd file")
end

"""Argument names documented for `fn`, read from the `\\arguments` block of the
Rd file that has `fn` as an alias; `nothing` if there is no such file."""
function arguments(dir, fn)
    man = joinpath(dir, "man")
    isdir(man) || return nothing
    for file in readdir(man; join = true)
        endswith(file, ".Rd") || continue
        text = read(file, String)
        occursin("\\alias{$fn}", text) || continue
        block = findfirst("\\arguments{", text)
        block === nothing && return String[]
        body, _ = braced(text, last(block))
        args = String[]
        pos = 1
        while (item = findnext("\\item{", body, pos)) !== nothing
            names, pos = braced(body, last(item))
            _, pos = braced(body, pos)
            for name in split(names, ',')
                name = strip(name)
                name in ("\\dots", "...") || push!(args, name)
            end
        end
        return args
    end
    return nothing
end

function check(package, spec)
    problems = String[]
    tag = latest_tag(spec["repo"])
    release = lstrip(tag, 'v')
    release == spec["release"] ||
        push!(problems, "latest release is $release, mapping was checked against $(spec["release"])")
    dir = fetch_source(spec["repo"], tag)
    available = exports(dir)
    mapped = spec["functions"]
    for fn in sort(collect(keys(mapped)))
        if !(fn in available)
            push!(problems, "`$fn` is no longer exported")
            continue
        end
        args = arguments(dir, fn)
        args === nothing && (push!(problems, "no help page found for `$fn`"); continue)
        for a in setdiff(mapped[fn], args)
            push!(problems, "`$fn` no longer has the argument `$a`")
        end
        for a in setdiff(args, mapped[fn])
            push!(problems, "`$fn` has an argument `$a` that the list does not have")
        end
    end
    for fn in setdiff(available, keys(mapped), get(spec, "unmapped", String[]))
        push!(problems, "`$fn` is exported but not in the list")
    end
    return release, problems
end

function main(selected)
    mapping = TOML.parsefile(MAPPING)
    packages = isempty(selected) ? sort(collect(keys(mapping))) : selected
    changed = false
    for package in packages
        haskey(mapping, package) || (println("$package: not in $MAPPING"); changed = true; continue)
        release, problems = check(package, mapping[package])
        if isempty(problems)
            println("$package $release: no changes")
        else
            changed = true
            println("$package $release:")
            foreach(p -> println("  - ", p), problems)
        end
    end
    return changed ? 1 : 0
end

exit(main(ARGS))
