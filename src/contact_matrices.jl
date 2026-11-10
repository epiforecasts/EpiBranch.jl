# ── Contact-matrix inputs ────────────────────────────────────────────
#
# Contact matrices come from surveys as arrays, often stratified by more than
# one variable (age and a socio-economic stratum, say). The functions here turn
# such an array into the flat types a multi-type model works with, give a
# `MixingProcess` its pair rate from a matrix and the group sizes, and build the
# array from a long-format table. Reading a table from a file is added by the
# package extensions for CSV and NetCDF. The format-specific part stays in the
# extensions to keep file-reader dependencies out of the core package.

"""
    StratifiedContacts

A contact array flattened into composite types, as returned by
[`flatten_contacts`](@ref) and [`contacts_from_table`](@ref).

  - `matrix`: the `N × N` contact matrix over the `N` composite types, laid out
    respondent by contact like any survey matrix: `matrix[j, i]` is the mean
    number of type-`i` contacts a type-`j` respondent makes per day.
  - `groups`: the composite type of each row and column, a tuple holding one
    level per stratifying variable, in the order of `names`.
  - `names`: the stratifying variables, as symbols.
  - `labels`: one string per composite type, such as `"child:home"`.

Use `matrix` and `labels` for the multi-type form of
[`BranchingProcess`](@ref), and `groups` with [`mixing_rate`](@ref) and a
[`MixingProcess`](@ref) with `mixing_by` set to `names`. The first stratifying
variable varies fastest down the types, so a population-size array of shape
`(n₁, n₂, …)` flattens to the matching vector with `vec`.
"""
struct StratifiedContacts
    matrix::Matrix{Float64}
    groups::Vector{Tuple}
    names::Tuple
    labels::Vector{String}
end

function Base.show(io::IO, c::StratifiedContacts)
    return print(
        io, "StratifiedContacts(", length(c.groups), " types by ", c.names, ")"
    )
end

"""
    flatten_contacts(contacts; names = nothing, levels = nothing)

Flatten a contact array stratified by one or more variables into a
[`StratifiedContacts`](@ref) whose types are the combinations of the variables'
levels.

`contacts` has the respondent axes of all variables first, then the contact
axes of all variables, the layout of socialmixr's multi-grouping
`compute_matrix` and of the contactmatrix class: for age (`n₁` bands) and a
stratum (`n₂` levels) the array is `n₁ × n₂ × n₁ × n₂` and
`contacts[a, s, b, t]` is the mean number of contacts per day a respondent of
age `a` in stratum `s` makes with people of age `b` in stratum `t`. One
variable gives the ordinary square matrix.

`names` are the variables' names as symbols (default `:stratum1`, `:stratum2`,
…), and `levels` a tuple with the levels of each variable (default `1:nᵥ`).
They set the composite types' `groups` and `labels`.

# Example

```julia
using EpiBranch, Distributions

# Two age bands by two settings; contacts[a, s, b, t].
contacts = rand(2, 2, 2, 2)
flat = flatten_contacts(contacts;
    names = (:age, :setting), levels = (["child", "adult"], ["home", "work"]))
flat.labels  # "child:home", "adult:home", "child:work", "adult:work"
BranchingProcess(flat.matrix, 0.05, 5.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5);
    type_labels = flat.labels)
```
"""
function flatten_contacts(
        contacts::AbstractArray{<:Real};
        names::Union{Nothing, Tuple} = nothing,
        levels::Union{Nothing, Tuple} = nothing
    )
    nd = ndims(contacts)
    (nd >= 2 && iseven(nd)) || throw(
        ArgumentError(
            "contacts needs a respondent axis and a contact axis per stratifying " *
                "variable, so an even number of axes; got $nd"
        )
    )
    k = nd ÷ 2
    sizes = ntuple(v -> size(contacts, v), k)
    for v in 1:k
        size(contacts, k + v) == sizes[v] || throw(
            ArgumentError(
                "axes $v and $(k + v) of contacts are the respondent and " *
                    "contact axes of variable $v and must have the same length, " *
                    "got $(size(contacts, v)) and $(size(contacts, k + v))"
            )
        )
    end
    names = something(names, ntuple(v -> Symbol(:stratum, v), k))
    length(names) == k || throw(
        ArgumentError("names has $(length(names)) entries for $k variables")
    )
    levels = something(levels, ntuple(v -> collect(1:sizes[v]), k))
    length(levels) == k || throw(
        ArgumentError("levels has $(length(levels)) entries for $k variables")
    )
    for v in 1:k
        length(levels[v]) == sizes[v] || throw(
            ArgumentError(
                "levels for variable $v has $(length(levels[v])) entries, " *
                    "expected $(sizes[v])"
            )
        )
        allunique(levels[v]) || throw(
            ArgumentError("levels for variable $v are not unique")
        )
    end

    N = prod(sizes)
    matrix = Matrix{Float64}(reshape(contacts, N, N))
    groups = Tuple[g for g in Iterators.product(levels...)]
    labels = String[join(string.(g), ":") for g in groups]
    return StratifiedContacts(matrix, vec(groups), Tuple(Symbol.(names)), vec(labels))
end

"""
    contacts_from_table(df; strata, value, names = nothing, levels = nothing,
                        fill = 0.0)

Build a [`StratifiedContacts`](@ref) from a long-format contact `DataFrame`,
one row per combination of respondent and contact levels.

`strata` lists one `respondent_column => contact_column` pair per stratifying
variable, and `value` names the column holding the mean number of contacts:

```julia
contacts_from_table(df;
    strata = [:part_age => :cont_age, :part_setting => :cont_setting],
    value = :contacts)
```

Each variable's levels are its distinct values in both columns, unless `levels`
gives them (a tuple of vectors, which also sets their order). A categorical
column keeps its own level order. Other columns are sorted naturally, with runs
of digits compared as numbers, so `"[5,15)"` comes before `"[15,Inf)"`. A
combination missing from the table is `fill`, `0.0` by default; one that
appears twice is an error, because adding the rows up is not always what a
long-format source means. `names` defaults to the respondent column names.
"""
function contacts_from_table(
        df::AbstractDataFrame; strata, value,
        names::Union{Nothing, Tuple} = nothing,
        levels::Union{Nothing, Tuple} = nothing,
        fill::Real = 0.0
    )
    pairs = [Symbol(p.first) => Symbol(p.second) for p in strata]
    isempty(pairs) && throw(ArgumentError("strata needs at least one pair of columns"))
    k = length(pairs)
    value = Symbol(value)
    for c in [value; first.(pairs); last.(pairs)]
        hasproperty(df, c) || throw(ArgumentError("the table has no column :$c"))
    end
    names = something(names, ntuple(v -> first(pairs[v]), k))
    if levels === nothing
        levels = ntuple(k) do v
            _table_levels(df[!, first(pairs[v])], df[!, last(pairs[v])])
        end
    end
    length(levels) == k || throw(
        ArgumentError("levels has $(length(levels)) entries for $k variables")
    )
    index = [Dict(l => i for (i, l) in enumerate(levels[v])) for v in 1:k]

    sizes = ntuple(v -> length(levels[v]), k)
    array = Base.fill(float(fill), sizes..., sizes...)
    seen = falses(size(array))
    for (row, r) in enumerate(eachrow(df))
        idx = ntuple(2k) do d
            v = d <= k ? d : d - k
            x = r[d <= k ? first(pairs[v]) : last(pairs[v])]
            haskey(index[v], x) || throw(
                ArgumentError("level $(repr(x)) of variable $(names[v]) is not in `levels`")
            )
            index[v][x]
        end
        seen[idx...] && throw(
            ArgumentError("row $row repeats a combination of levels already in the table")
        )
        seen[idx...] = true
        ismissing(r[value]) && throw(ArgumentError("row $row has a missing $value"))
        array[idx...] = r[value]
    end
    return flatten_contacts(array; names, levels)
end

# Sort key for natural order: runs of digits compare as numbers, other text as
# text.
function _natural_key(x)
    return [
        m.captures[1] === nothing ? (1, m.match, 0) : (0, "", parse(BigInt, m.match))
            for m in eachmatch(r"(\d+)|(\D+)", string(x))
    ]
end

# The levels of one variable from its respondent and contact columns. A
# categorical column keeps its own level order, the contact column's extra
# levels following; anything else is sorted naturally.
function _table_levels(respondent, contact)
    if DataAPI.refpool(respondent) !== nothing
        found = collect(DataAPI.levels(respondent))
        append!(found, DataAPI.refpool(contact) !== nothing ? DataAPI.levels(contact) : skipmissing(contact))
        return unique(found)
    end
    found = unique(vcat(collect(skipmissing(respondent)), collect(skipmissing(contact))))
    return sort!(found; by = _natural_key)
end

"""
    ContactMatrixRate

The pair rate [`mixing_rate`](@ref) returns: a callable
`rate(infector_group, target_group)`.
"""
struct ContactMatrixRate
    rates::Matrix{Float64}
    index::Dict{Any, Int}
end

function (r::ContactMatrixRate)(from, to)
    haskey(r.index, from) || throw(ArgumentError("unknown mixing group $(repr(from))"))
    haskey(r.index, to) || throw(ArgumentError("unknown mixing group $(repr(to))"))
    return r.rates[r.index[from], r.index[to]]
end

"""
    mixing_rate(contacts, group_sizes; groups = nothing)
    mixing_rate(contacts::StratifiedContacts, group_sizes)

The `rate` of a [`MixingProcess`](@ref) from a contact matrix and the sizes of
the groups. `contacts[j, i]` is the mean number of group-`i` contacts a
group-`j` person makes per day, the layout of a survey matrix and of the
matrix [`BranchingProcess`](@ref) takes. A person in group `j` then contacts
one given member of group `i` at rate `contacts[j, i] / group_sizes[i]`, the
value `rate(j, i)` returns; summed over the `group_sizes[i]` members of group
`i` it gives back `contacts[j, i]`.

`group_sizes` holds the number of people in each group, all positive. A group
is the tuple of attribute values `MixingProcess` reads with `mixing_by`. Without
`groups`, group `i` is `(i,)`, for a `mixing_by` of one attribute taking the
values `1:n`. Pass `groups`, a vector of tuples in the matrix's order, for other
levels or for several variables; a [`StratifiedContacts`](@ref) supplies its
own, and a population-size array of shape `(n₁, n₂, …)` is flattened to match
its types.

The matrix is used as given and is not made reciprocal
(`contacts[j, i] * n[j] == contacts[i, j] * n[i]`). A matrix that is not
reciprocal gives a pair rate that depends on the direction.

# Example

```julia
using EpiBranch, Distributions

n = [1500, 1500]
contacts = [3.0 0.5; 0.5 0.5]
band = (rng, ind) -> (ind.state[:age_band] = rand(rng, 1:2))
model = ModelSpec(
    MixingProcess(; population_size = sum(n), mixing_by = (:age_band,),
        rate = mixing_rate(contacts, n));
    progression = [Transition(:recovered; from = :infection, rate = 1.0, terminal = true)],
    attributes = band)
```
"""
function mixing_rate(
        contacts::AbstractMatrix{<:Real}, group_sizes::AbstractVector{<:Real};
        groups::Union{Nothing, AbstractVector} = nothing
    )
    n = length(group_sizes)
    size(contacts) == (n, n) || throw(
        ArgumentError(
            "contacts is $(size(contacts, 1)) × $(size(contacts, 2)) but " *
                "group_sizes has $n entries"
        )
    )
    all(>(0), group_sizes) || throw(ArgumentError("group_sizes must all be positive"))
    all(>=(0), contacts) || throw(ArgumentError("contacts must not be negative"))
    groups = groups === nothing ? [(i,) for i in 1:n] : groups
    length(groups) == n || throw(
        ArgumentError("groups has $(length(groups)) entries for $n groups")
    )
    allunique(groups) || throw(ArgumentError("groups are not unique"))
    rates = Matrix{Float64}(contacts) ./ permutedims(Float64.(group_sizes))
    return ContactMatrixRate(rates, Dict{Any, Int}(g => i for (i, g) in enumerate(groups)))
end

function mixing_rate(contacts::StratifiedContacts, group_sizes::AbstractArray{<:Real})
    return mixing_rate(contacts.matrix, vec(group_sizes); groups = contacts.groups)
end

# ── Reading contact matrices from files ──────────────────────────────

"""
    ContactFileFormat

Abstract supertype of the file formats [`read_contacts`](@ref) reads:
[`CSVFormat`](@ref), which needs the `CSV` package loaded, and
[`NetCDFFormat`](@ref), which needs `NCDatasets`. Loading the package
activates the extension that holds the reader.
"""
abstract type ContactFileFormat end

"Long-format CSV contact matrices, read once `CSV` is loaded."
struct CSVFormat <: ContactFileFormat end

"NetCDF contact arrays, read once `NCDatasets` is loaded."
struct NetCDFFormat <: ContactFileFormat end

function _format_from_path(path::AbstractString)
    ext = lowercase(last(splitext(path)))
    ext in (".csv", ".tsv", ".txt") && return CSVFormat()
    ext in (".nc", ".nc4", ".cdf") && return NetCDFFormat()
    throw(
        ArgumentError(
            "cannot tell the contact-matrix format of $(repr(path)) from its " *
                "extension; pass `format = CSVFormat()` or `format = NetCDFFormat()`"
        )
    )
end

_format_package(::CSVFormat) = "CSV"
_format_package(::NetCDFFormat) = "NCDatasets"
_format_package(::ContactFileFormat) = "the package for the format"

"""
    read_contacts(path; format = <from the extension>, kwargs...)

Read a contact matrix from a file into a [`StratifiedContacts`](@ref).

  - **CSV** (`.csv`, needs `using CSV`): a long-format table, one row per
    respondent-by-contact combination, read as for
    [`contacts_from_table`](@ref) with its `strata`, `value`, `names`, `levels`
    and `fill` keywords. `delim` sets the delimiter, a comma by default.
  - **NetCDF** (`.nc`, needs `using NCDatasets`): the array variable `variable`,
    with `strata` giving, for each stratifying variable, the names of its
    respondent and contact dimensions as a `"part_age" => "cont_age"` pair.
    Levels come from a coordinate variable named like the dimension when the
    file has one, and are `1:n` otherwise. `names` and `levels` are as for
    [`flatten_contacts`](@ref).

Pass `format = CSVFormat()` or `NetCDFFormat()` for a file whose extension does
not say.

# Example

```julia
using EpiBranch, CSV, Distributions

flat = read_contacts("contacts.csv";
    strata = ["part_age" => "cont_age"], value = "contacts")
BranchingProcess(flat.matrix, 0.05, 5.0, R -> NegBin(R, 0.5), LogNormal(1.6, 0.5);
    type_labels = flat.labels)
```
"""
function read_contacts(
        path::AbstractString;
        format::ContactFileFormat = _format_from_path(path), kwargs...
    )
    return read_contacts(format, path; kwargs...)
end

function read_contacts(format::ContactFileFormat, path::AbstractString; kwargs...)
    throw(
        ArgumentError(
            "reading $(nameof(typeof(format))) contact matrices needs " *
                "`using $(_format_package(format))` first"
        )
    )
end
