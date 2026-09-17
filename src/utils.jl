"""
    scale_distribution(d, factor::Real)

Scale an offspring distribution's mean by `factor`, preserving its family and
shape. Defined for the two offspring families the package scales: `Poisson`
(returns `Poisson(λ · factor)`) and `NegativeBinomial` (same `k`, mean scaled).
Any other family raises an `ArgumentError` naming it, rather than a bare
`MethodError`.
"""
function scale_distribution(d::Poisson, factor::Real)
    Poisson(mean(d) * factor)
end

function scale_distribution(d::NegativeBinomial, factor::Real)
    k = d.r
    new_mean = mean(d) * factor
    p = k / (k + new_mean)
    NegativeBinomial(k, p)
end

function scale_distribution(d::Distribution, ::Real)
    throw(ArgumentError(
        "scale_distribution is not defined for $(typeof(d)); only Poisson and " *
        "NegativeBinomial offspring distributions can be scaled."))
end

# A minimal binary min-heap over `(time, id, ...)` tuples, used by the
# continuous-time event loops. Tuples compare lexicographically, so ties break
# deterministically on the integers that follow the time: the member id, and on
# the race the proposal's own number, which orders two proposals made to one
# member at the same time by which was made first. Keeping the heap here avoids a
# dependency on DataStructures.
const _HeapEntry = Tuple{<:Real, Vararg{Int}}

function _heap_push!(h::Vector{E}, x::E) where {E <: _HeapEntry}
    push!(h, x)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        h[p] <= h[i] && break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return h
end

# What an empty heap reports: an event that never happens, with zero ids.
function _heap_empty(::Type{E}) where {E <: _HeapEntry}
    ntuple(i -> i == 1 ? fieldtype(E, 1)(Inf) : 0, Val(fieldcount(E)))::E
end

function _heap_peek(h::Vector{E}) where {E <: _HeapEntry}
    isempty(h) ? _heap_empty(E) : @inbounds h[1]
end

function _heap_pop!(h::Vector{E}) where {E <: _HeapEntry}
    n = length(h)
    n == 0 && return _heap_empty(E)
    @inbounds top = h[1]
    @inbounds last = h[n]
    pop!(h)
    n -= 1
    if n > 0
        @inbounds h[1] = last
        i = 1
        @inbounds while true
            l = 2i
            r = 2i + 1
            s = i
            (l <= n && h[l] < h[s]) && (s = l)
            (r <= n && h[r] < h[s]) && (s = r)
            s == i && break
            h[i], h[s] = h[s], h[i]
            i = s
        end
    end
    return top
end
