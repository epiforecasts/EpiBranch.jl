# ── Event queue ────────────────────────────────────────────────────────
# The continuous-time engine works through one queue of events in time order.
# An event is a plain immutable value, so the queue holds them inline and a pop
# allocates nothing and dispatches on nothing.

# One event: when it falls due, which member it is about (a position in the
# race's `members`), and which of that member's proposals it stands for.
struct _Event{T <: Real}
    time::T
    member::Int
    ref::Int
end

# Events fall due in time order. Ties break on the member, then on the
# proposal's own number, which orders two proposals made to one member at the
# same time by which was made first, so the order never depends on how the
# queue happened to be built.
function Base.isless(a::_Event, b::_Event)
    a.time == b.time || return isless(a.time, b.time)
    a.member == b.member || return a.member < b.member
    return a.ref < b.ref
end

# The queue starts as a binary min-heap. Once it holds more events than this it
# moves them into a radix heap, whose cost per event does not grow with its size;
# a binary heap of hundreds of thousands of events spends most of its time on
# cache misses. Small queues, such as one per household or per small outbreak,
# never pay for the radix heap's buckets.
const _RADIX_THRESHOLD = 1024

# A radix heap orders events by an unsigned key that sorts like their time. Only
# `Float64` times get one; other time types (a dual number under automatic
# differentiation) stay on the binary heap.
_radix_capable(::Type{Float64}) = true
_radix_capable(::Type) = false

# The bits of a non-negative time with the sign bit set, and of a negative time
# with every bit flipped, order as the times do, negative ones included. Adding
# zero turns a negative zero into a positive one first.
@inline function _time_key(t::Float64)
    bits = reinterpret(UInt64, t + 0.0)
    return signbit(t + 0.0) ? ~bits : bits | (UInt64(1) << 63)
end

# Bucket 1 holds the events at the time last popped, as a binary heap ordered
# by member and proposal: a large outbreak seeds every case at one time, and a
# sorted vector would make those ties quadratic. Bucket `b > 1` holds events whose
# time first differs from the last popped in bit `b - 2`, counted from the
# lowest.
@inline _bucket(key::UInt64, last::UInt64) =
    key == last ? 1 : 65 - leading_zeros(key ⊻ last)

mutable struct _EventQueue{T <: Real}
    heap::Vector{_Event{T}}
    buckets::Vector{Vector{_Event{T}}}
    radix::Bool
    last::UInt64
    n::Int
end
function _EventQueue{T}() where {T}
    return _EventQueue{T}(_Event{T}[], Vector{_Event{T}}[], false, UInt64(0), 0)
end

_radix(q::_EventQueue) = q.radix

Base.isempty(q::_EventQueue) = q.n == 0

# An emptied queue starts again on the binary heap, but keeps its buckets and
# the room they grew to, so a queue used again does not grow them afresh.
function Base.empty!(q::_EventQueue)
    empty!(q.heap)
    foreach(empty!, q.buckets)
    q.radix = false
    q.last = UInt64(0)
    q.n = 0
    return q
end


function _enqueue!(q::_EventQueue{T}, e::_Event{T}) where {T}
    q.n += 1
    if _radix(q)
        _radix_push!(q, e)
    else
        _heap_push!(q.heap, e)
        _radix_capable(T) && q.n > _RADIX_THRESHOLD && _to_radix!(q)
    end
    return q
end

# The earliest event, removed from the queue. Only called on a queue that is not
# empty.
function _dequeue!(q::_EventQueue)
    q.n -= 1
    _radix(q) && return _radix_pop!(q)
    e = _heap_pop!(q.heap)
    _radix_capable(typeof(e.time)) && (q.last = _time_key(e.time))
    return e
end

# Every event still in `q`, visited once each without removing it, wherever it
# currently sits: the binary heap, or, once migrated, the radix buckets. Used
# only where an operation otherwise discards the whole queue and must first
# save the events it does not itself know how to rebuild (see
# `_compact_proposals!`).
function _each_event(f, q::_EventQueue)
    foreach(f, q.heap)
    foreach(bucket -> foreach(f, bucket), q.buckets)
    return nothing
end

function _heap_push!(h::Vector{E}, e::E) where {E <: _Event}
    push!(h, e)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        isless(h[i], h[p]) || break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return h
end

function _heap_pop!(h::Vector{<:_Event})
    n = length(h)
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
            (l <= n && isless(h[l], h[s])) && (s = l)
            (r <= n && isless(h[r], h[s])) && (s = r)
            s == i && break
            h[i], h[s] = h[s], h[i]
            i = s
        end
    end
    return top
end

# Every queued event falls due at or after the last popped, since a proposal is
# never made for a time before the clock; the radix heap relies on it.
function _to_radix!(q::_EventQueue{T}) where {T}
    isempty(q.buckets) && (q.buckets = [_Event{T}[] for _ in 1:65])
    q.radix = true
    for e in q.heap
        _radix_push!(q, e)
    end
    empty!(q.heap)
    return q
end

function _radix_push!(q::_EventQueue, e::_Event)
    key = _time_key(e.time)
    key < q.last && throw(
        ArgumentError(
            "an event at time $(e.time) was queued before the last event taken from the queue"
        )
    )
    b = _bucket(key, q.last)
    b == 1 ? _heap_push!(q.buckets[1], e) : push!(q.buckets[b], e)
    return q
end

function _radix_pop!(q::_EventQueue)
    current = q.buckets[1]
    if isempty(current)
        b = 2
        @inbounds while isempty(q.buckets[b])
            b += 1
        end
        source = q.buckets[b]
        q.last = minimum(e -> _time_key(e.time), source)
        # Every event moves to a lower bucket, relative to the new last time.
        for e in source
            b = _bucket(_time_key(e.time), q.last)
            b == 1 ? _heap_push!(current, e) : push!(q.buckets[b], e)
        end
        empty!(source)
    end
    return _heap_pop!(current)
end
