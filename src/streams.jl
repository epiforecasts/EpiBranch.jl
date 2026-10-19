# ── Common random numbers ────────────────────────────────────────────
#
# Every draw the engine makes comes from one `rng`, in whatever order it
# needs them, so a draw with no effect on the outbreak still shifts every
# draw the engine asks for afterwards. Two scenarios built from the same seed
# then share nothing once they first differ, instead of sharing everything
# up to the point one of them changed. `stream` is the seam that lets a draw
# which belongs to one person, or one pair of them, come from a place of its
# own instead: by default it is the one shared sequence, unchanged, and
# `CommonRandomNumbers` opts a run into keeping its own place per `(id,
# purpose)` so that later draws for other people do not move it.

"""
    stream(rng, id, purpose) -> AbstractRNG

The generator to draw from for `id` (a person's `id`, or `(a, b)` for an
ordered pair of them) and `purpose` (a `Symbol` or tag telling apart the
different things drawn for the same `id`, such as `:susceptibility` and
`:infectiousness` for one person, or `:contact` for one pair). The default
method, for any ordinary `AbstractRNG`, returns `rng` itself: every draw still
comes from the one sequence, in the order the engine asks for it, exactly as
before `stream` existed.

[`CommonRandomNumbers`](@ref) is the opt-in alternative: it returns a
generator kept just for `(id, purpose)`, so a run built from it draws the
same values for that `id` and `purpose` whatever else in the run has changed.
A new transmission model or intervention that draws its own per-person or
per-pair values should call `stream(rng, id, purpose)` for the generator to
draw from, rather than `rng` directly, to take part.
"""
stream(rng::Random.AbstractRNG, id, purpose) = rng

"""
    CommonRandomNumbers(seed)

An `AbstractRNG`, built from `seed`, that keeps two runs on the same draws for
everything routed through [`stream`](@ref) — progression delays,
infectiousness and susceptibility, acceptance of an offer, and a pair's
contact times — until one of them changes what actually happens to a
particular person or pair. Pass it as `simulate`'s `rng` in place of a plain
seeded generator to compare scenarios on shared random numbers:

```julia
baseline = simulate(ModelSpec(net; progression); rng = CommonRandomNumbers(12))
compared = simulate(
    ModelSpec(net; progression, interventions); rng = CommonRandomNumbers(12)
)
```

A draw not routed through `stream` — an offspring count, which node is seeded
first, anything a custom attribute or model reads from `rng` directly — still
comes from one shared sequence, exactly as for any other `AbstractRNG`, so
running the same model twice with the same `CommonRandomNumbers` reproduces
it exactly either way.
"""
struct CommonRandomNumbers{R <: Random.AbstractRNG} <: Random.AbstractRNG
    seed::UInt64
    rng::R
    substreams::Dict{Any, Random.Xoshiro}
end

function CommonRandomNumbers(seed::Integer)
    s = UInt64(seed)
    return CommonRandomNumbers(s, Random.Xoshiro(s), Dict{Any, Random.Xoshiro}())
end

# The one primitive an `AbstractRNG` needs for everything else — `rand` for
# any type, `randn`, `randexp`, ranges, arrays — to work generically off it.
Random.rng_native_52(c::CommonRandomNumbers) = Random.rng_native_52(c.rng)
Random.rand(c::CommonRandomNumbers, sp::Random.SamplerType{UInt64}) = rand(c.rng, sp)

# A fresh generator per `(id, purpose)`, seeded independently of the shared
# sequence and of each other, so creating one costs nothing from `c.rng` and
# asking for it twice (a pair resampled after a blocked contact, a step
# revisited) carries on from where it left off rather than repeating.
function stream(c::CommonRandomNumbers, id, purpose)
    return get!(c.substreams, (id, purpose)) do
        Random.Xoshiro(hash((c.seed, id, purpose)))
    end
end
