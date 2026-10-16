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
# same time by which was made first, so the order never depends on how the heap
# happened to be built.
function Base.isless(a::_Event, b::_Event)
    a.time == b.time || return isless(a.time, b.time)
    a.member == b.member || return a.member < b.member
    return a.ref < b.ref
end

# A binary min-heap of events. Kept here rather than taken from
# DataStructures, which would be a dependency for one small structure.
struct _EventQueue{T <: Real}
    heap::Vector{_Event{T}}
end
_EventQueue{T}() where {T} = _EventQueue{T}(_Event{T}[])

Base.isempty(q::_EventQueue) = isempty(q.heap)
Base.empty!(q::_EventQueue) = (empty!(q.heap); q)

function _enqueue!(q::_EventQueue{T}, e::_Event{T}) where {T}
    h = q.heap
    push!(h, e)
    i = length(h)
    @inbounds while i > 1
        p = i >> 1
        isless(h[i], h[p]) || break
        h[p], h[i] = h[i], h[p]
        i = p
    end
    return q
end

# The earliest event, removed from the queue. Only called on a queue that is not
# empty.
function _dequeue!(q::_EventQueue)
    h = q.heap
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
