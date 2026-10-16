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

# A radix heap orders events by the bit pattern of their time, which sorts like
# the time itself for the non-negative `Float64` times the race uses. Only those
# have such a pattern; other time types (a dual number under automatic
# differentiation) stay on the binary heap.
_radix_capable(::Type{Float64}) = true
_radix_capable(::Type) = false

# Times are compared through their bits once in the radix heap. Adding zero
# turns a negative zero into a positive one, whose pattern is the smallest.
@inline _time_key(t::Float64) = reinterpret(UInt64, t + 0.0)

# Bucket 1 holds the events at the time last popped, kept sorted so the earliest
# by member and proposal comes off the end. Bucket `b > 1` holds events whose
# time first differs from the last popped in bit `b - 2`, counted from the
# lowest.
@inline _bucket(key::UInt64, last::UInt64) =
    key == last ? 1 : 65 - leading_zeros(key ⊻ last)

mutable struct _EventQueue{T <: Real}
    heap::Vector{_Event{T}}
    buckets::Vector{Vector{_Event{T}}}
    last::UInt64
    n::Int
end
_EventQueue{T}() where {T} = _EventQueue{T}(_Event{T}[], Vector{_Event{T}}[], UInt64(0), 0)

_radix(q::_EventQueue) = !isempty(q.buckets)

Base.isempty(q::_EventQueue) = q.n == 0

function Base.empty!(q::_EventQueue)
    empty!(q.heap)
    foreach(empty!, q.buckets)
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
    q.buckets = [_Event{T}[] for _ in 1:65]
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
    bucket = q.buckets[b]
    push!(bucket, e)
    if b == 1
        # Keep the current time's events sorted with the earliest last.
        i = length(bucket)
        @inbounds while i > 1 && isless(bucket[i - 1], e)
            bucket[i] = bucket[i - 1]
            i -= 1
        end
        @inbounds bucket[i] = e
    end
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
            push!(q.buckets[_bucket(_time_key(e.time), q.last)], e)
        end
        empty!(source)
        sort!(current; rev = true)
    end
    return pop!(current)
end
