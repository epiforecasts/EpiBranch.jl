# Formats a date can be written in besides ISO 8601, which is what `Date`
# prints by default. Day-first and month-first forms are both here because
# `03/04/2024` is the ambiguity cleaning code most often gets wrong.
const _MESSY_DATE_FORMATS = (
    dateformat"dd/mm/yyyy", dateformat"mm/dd/yyyy", dateformat"d u yyyy",
    dateformat"yyyy/mm/dd", dateformat"dd-mm-yy",
)

function _check_rate(name::Symbol, rate)
    0 <= rate <= 1 || throw(ArgumentError("`$name` must be between 0 and 1, got $rate."))
    return nothing
end

"""Misspell a category: swap two neighbouring letters, drop one, change the
case, or pad with a space."""
function _misspell(rng::AbstractRNG, s::AbstractString)
    n = length(s)
    n == 0 && return String(s)
    chars = collect(s)
    kind = rand(rng, 1:4)
    if kind == 1 && n >= 2
        i = rand(rng, 1:(n - 1))
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        out = String(chars)
        # A swap of two equal letters changes nothing, so drop one instead.
        return out == s ? String(chars[1:(n - 1)]) : out
    elseif kind == 2 && n >= 2
        deleteat!(chars, rand(rng, 1:n))
        return String(chars)
    elseif kind == 3
        out = s == uppercase(s) ? lowercase(s) : uppercase(s)
        return out == s ? string(s, " ") : out
    end
    return string(s, " ")
end

function _is_date_column(col)
    return any(x -> x isa Date, col)
end

_is_text_column(col) = any(x -> x isa Union{AbstractString, Symbol}, col)

"""
    messy_linelist(df::DataFrame; rng = Random.default_rng(), kwargs...)

Make a line list messy, so that cleaning code can be tested against data whose
true values are known. Takes the table returned by [`linelist`](@ref) (or any
`DataFrame` laid out like it) and returns a copy with the kinds of mess found
in real line lists. It works on the table alone and never touches the
simulation. The same `rng` gives the same result.

Each kind of mess has its own rate, a probability between 0 and 1, and a rate
of 0 leaves that kind out:

- `missing_rate`: the chance that a value is replaced by `missing`.
- `implausible_rate`: the chance that a value is replaced by one that cannot be
  true. A date moves 20 to 60 years away from its true value, and a number
  becomes negative or 100 times larger.
- `date_format_rate`: the chance that a date is written in another format
  (`dd/mm/yyyy`, `mm/dd/yyyy`, `d u yyyy`, `yyyy/mm/dd` or `dd-mm-yy`) rather than
  `yyyy-mm-dd`. Dates become text as soon as this rate is above 0.
- `typo_rate`: the chance that a text or `Symbol` value is misspelt, written
  in the wrong case or padded with a space. Such a column becomes text.
- `duplicate_rate`: the chance that a row is repeated, with the same values as
  the messy original. Duplicates are added at the end.

Columns in `protect` are left alone by every kind of mess, by default the
identifying columns of a line list: `:id`, `:parent_id`, `:generation` and
`:chain_id`. The rows of the clean table can therefore be matched to those of
the messy one by `id`. A value gets at most one kind of mess other than
duplication. Booleans and any value that is not a date, number, text or
`Symbol` are never made implausible.

# Examples

```julia
using Dates, Random
state = simulate(ModelSpec(BranchingProcess(NegBin(2.5, 0.16), Gamma(2.0, 3.0)));
    max_cases = 100)
clean = linelist(state; reference_date = Date(2024, 1, 1))
messy = messy_linelist(clean; rng = Xoshiro(1), missing_rate = 0.05)
```
"""
function messy_linelist(
        df::DataFrame; rng::AbstractRNG = Random.default_rng(),
        missing_rate::Real = 0.05, implausible_rate::Real = 0.02,
        date_format_rate::Real = 0.2, typo_rate::Real = 0.05,
        duplicate_rate::Real = 0.03,
        protect = (:id, :parent_id, :generation, :chain_id)
    )
    for (name, rate) in (
            (:missing_rate, missing_rate), (:implausible_rate, implausible_rate),
            (:date_format_rate, date_format_rate), (:typo_rate, typo_rate),
            (:duplicate_rate, duplicate_rate),
        )
        _check_rate(name, rate)
    end
    protected = Set(Symbol.(protect))
    n = nrow(df)
    out = DataFrame()
    for name in propertynames(df)
        col = df[!, name]
        if name in protected
            out[!, name] = copy(col)
            continue
        end
        out[!, name] = _messy_column(
            rng, col, missing_rate, implausible_rate, date_format_rate, typo_rate
        )
    end
    duplicated = [i for i in 1:n if rand(rng) < duplicate_rate]
    return isempty(duplicated) ? out : vcat(out, out[duplicated, :])
end

function _messy_column(rng, col, missing_rate, implausible_rate, date_format_rate, typo_rate)
    is_date = _is_date_column(col)
    is_text = _is_text_column(col)
    # A date column becomes text only when date formats are mixed, and a
    # category column only when typos are allowed. A rate of 0 keeps the type.
    to_text = (is_date && date_format_rate > 0) || (is_text && typo_rate > 0)
    values = Vector{Any}(undef, length(col))
    for (i, x) in enumerate(col)
        if ismissing(x)
            values[i] = missing
        elseif rand(rng) < missing_rate
            values[i] = missing
        elseif rand(rng) < implausible_rate
            values[i] = _implausible(rng, x)
        elseif x isa Date && date_format_rate > 0
            fmt = rand(rng) < date_format_rate ? rand(rng, _MESSY_DATE_FORMATS) :
                dateformat"yyyy-mm-dd"
            values[i] = Dates.format(x, fmt)
        elseif x isa Union{AbstractString, Symbol} && rand(rng) < typo_rate
            values[i] = _misspell(rng, String(x))
        elseif x isa Symbol && to_text
            values[i] = String(x)
        else
            values[i] = x
        end
    end
    if to_text
        # An implausible date in a column that is turned into text has to be
        # written out like the rest.
        for i in eachindex(values)
            values[i] isa Date && (values[i] = Dates.format(values[i], dateformat"yyyy-mm-dd"))
            values[i] isa Symbol && (values[i] = String(values[i]))
        end
    end
    kinds = Set(typeof(v) for v in values if !ismissing(v))
    T = isempty(kinds) ? Missing : Union{Missing, typejoin(kinds...)}
    return convert(Vector{T}, values)
end

function _implausible(rng::AbstractRNG, x::Date)
    years = rand(rng, 20:60) * rand(rng, (-1, 1))
    return x + Year(years)
end
_implausible(::AbstractRNG, x::Bool) = x
function _implausible(rng::AbstractRNG, x::Real)
    return rand(rng, Bool) ? -abs(x) - 1 : 100 * x
end
_implausible(::AbstractRNG, x) = x
