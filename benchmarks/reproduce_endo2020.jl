#!/usr/bin/env julia
# Reproduce the Endo, Abbott, Kucharski & Funk 2020 analysis on the
# WHO 27-Feb-2020 situation report data, comparing Endo's threshold-
# based likelihood with the analytical real-time end-of-outbreak
# mixture on real per-country case timing.
#
# Data:
# - benchmarks/endo2020_who_27feb2020.csv (43 countries, columns
#   Total / ImportedChina / ImportedOthers / Local / Unknown / Death).
# - benchmarks/endo2020_dailycases_27feb2020.csv (per-day new case
#   counts for each country up to the cutoff).
# Endo's published analysis excluded Egypt and Iran (zero imported).

using EpiBranch
using DataFrames
using Dates
using Distributions
using StableRNGs

# Endo et al.'s SARS-CoV-2 generation time (LogNormal mean ≈ 4.7 d, sd ≈ 2.9 d).
# A Gamma fit with mean 5, sd ≈ 3 is close enough for a comparison.
const GT = Gamma(2.78, 1.8)         # mean 5, sd 3
const CUTOFF = Date(2020, 2, 27)
const DATA_PATH = joinpath(@__DIR__, "endo2020_who_27feb2020.csv")
const DAILY_PATH = joinpath(@__DIR__, "endo2020_dailycases_27feb2020.csv")

const EXCLUDED_COUNTRIES = ["Egypt", "Iran"]

function load_data()
    df = DataFrame()
    open(DATA_PATH) do io
        # Read header (note BOM in original).
        header = split(replace(readline(io), "\ufeff" => ""), ",")
        cols = ["Country", header[2:end]...]
        for col in cols
            df[!, Symbol(col)] = String[]
        end
        for line in eachline(io)
            fields = split(line, ",")
            length(fields) == length(cols) || continue
            push!(df, fields)
        end
    end
    df.Total = parse.(Int, df.Total)
    df.ImportedChina = parse.(Int, df.ImportedChina)
    df.ImportedOthers = parse.(Int, df.ImportedOthers)
    return df
end

function endo_subset(df)
    seeds_total = df.ImportedChina .+ df.ImportedOthers
    keep = (.!in.(df.Country, Ref(EXCLUDED_COUNTRIES))) .&
           (seeds_total .>= 1) .& (df.Total .>= seeds_total)
    sub = df[keep, :]
    sizes = sub.Total
    seeds = sub.ImportedChina .+ sub.ImportedOthers
    return sub.Country, sizes, seeds
end

"""
Parse the dailycases CSV and return a Dict mapping country name to a
`(first_case_date, last_case_date)` tuple, restricted to dates ≤ cutoff.
The file's row 4 is the date header; subsequent rows are countries with
daily new cases prefixed by `+`. Empty cells mean zero new cases.
"""
function load_daily_dates()
    lines = readlines(DAILY_PATH)
    # Row 4 (1-indexed) carries the dates from column 3 onwards.
    header_fields = _split_csv_row(lines[4])
    date_cells = header_fields[3:end]
    dates = Date[]
    for cell in date_cells
        s = strip(cell)
        isempty(s) && break
        push!(dates, _parse_endo_date(s))
    end
    out = Dict{String, Tuple{Date, Date}}()
    for line in lines[5:end]
        fields = _split_csv_row(line)
        length(fields) < 3 && continue
        country = strip(fields[2])
        (isempty(country) || startswith(country, "@")) && continue
        first_d, last_d = nothing, nothing
        for (j, cell) in enumerate(fields[3:end])
            j > length(dates) && break
            n = _parse_endo_count(cell)
            n <= 0 && continue
            d = dates[j]
            d > CUTOFF && continue
            first_d === nothing && (first_d = d)
            last_d = d
        end
        first_d !== nothing && (out[country] = (first_d, last_d))
    end
    return out
end

# Quoted-comma-aware CSV splitter: strings like "1,779" stay intact.
function _split_csv_row(line::AbstractString)
    fields = String[]
    buf = IOBuffer()
    in_quote = false
    for c in line
        if c == '"'
            in_quote = !in_quote
        elseif c == ',' && !in_quote
            push!(fields, String(take!(buf)))
        else
            print(buf, c)
        end
    end
    push!(fields, String(take!(buf)))
    return fields
end

function _parse_endo_date(s::AbstractString)
    s = strip(replace(s, '\u00a0' => ' '))   # NBSP
    # E.g. "Jan 13, 2020" or " Mar 4, 2020  11:44 AM GMT" — take first 3 tokens.
    tokens = split(s)
    parse_str = tokens[1] * " " * replace(tokens[2], "," => "") * " " *
                tokens[3]
    return Date(parse_str, dateformat"u d yyyy")
end

function _parse_endo_count(cell::AbstractString)
    s = strip(replace(cell, "," => ""))
    isempty(s) && return 0
    s = replace(s, "+" => "", "-" => "")
    isempty(s) && return 0
    return something(tryparse(Int, s), 0)
end

# Grid for MLE search.
const R_GRID = 0.1:0.1:5.0
const K_GRID = 0.05:0.05:2.0

function grid_mle(loglik_fn; R_grid = R_GRID, K_grid = K_GRID)
    best_R, best_k, best_ll = first(R_grid), first(K_grid), -Inf
    for R in R_grid, k in K_grid

        ll = loglik_fn(R, k)
        if ll > best_ll
            best_R, best_k, best_ll = R, k, ll
        end
    end
    return (R = best_R, k = best_k, ll = best_ll)
end

function main()
    df = load_data()
    countries, sizes, seeds = endo_subset(df)
    daily = load_daily_dates()
    # Some countries in the bycountries CSV don't appear in the daily
    # CSV (e.g. naming differences). Drop those — Endo would have done
    # the same when applying the 7-day rule.
    has_dates = [haskey(daily, c) for c in countries]
    countries = countries[has_dates]
    sizes = sizes[has_dates]
    seeds = seeds[has_dates]
    cluster_ages = [Float64(Dates.value(CUTOFF - daily[c][1])) for c in countries]
    taus = [Float64(Dates.value(CUTOFF - daily[c][2])) for c in countries]

    println("Loaded $(length(sizes)) countries with daily case timing " *
            "(excluded: " * join(EXCLUDED_COUNTRIES, ", ") * "; " *
            "$(count(.!has_dates)) dropped for missing daily data)")
    println("Total cases:  ", sum(sizes))
    println("Total imports:", sum(seeds))
    println("τ range:      $(minimum(taus))–$(maximum(taus)) days; " *
            "$(count(>=(7), taus))/$(length(taus)) silent ≥7 d")
    println("cluster_age range: $(minimum(cluster_ages))–" *
            "$(maximum(cluster_ages)) days")
    println()

    # Threshold rule: 7-day silence as in Endo.
    concluded = taus .>= 7.0
    chain_data = ChainSizes(sizes; seeds = seeds, concluded = concluded)
    threshold_loglik = (R, k) -> loglikelihood(chain_data, NegBin(R, k))

    # Analytical real-time likelihood with real per-country τ.
    rt_data = RealTimeChainSizes(sizes, taus; seeds = seeds)
    rt_analytical_loglik = function (R, k)
        model = BranchingProcess(NegBin(R, k), GT)
        return loglikelihood(rt_data, model)
    end

    methods = [
        ("Endo threshold (7-day rule)", threshold_loglik),
        ("RT analytical (true τ)", rt_analytical_loglik)]

    println("Method                          | MLE R   MLE k   LL@MLE")
    println("―" ^ 65)
    for (name, fn) in methods
        mle = grid_mle(fn)
        println(rpad(name, 32) * "| " *
                rpad(string(round(mle.R, digits = 2)), 8) *
                rpad(string(round(mle.k, digits = 2)), 8) *
                string(round(mle.ll, digits = 2)))
    end
    println()
    println("Endo et al. 2020 reported (joint, threshold rule):")
    println("  R₀ ≈ 1.4 – 12 (95% CrI), k ≈ 0.04 – 0.2 (95% CrI)")
    println("  Median k ≈ 0.1 with R₀ fixed at 2.5.")
end

main()
