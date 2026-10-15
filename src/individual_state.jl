# ── Per-individual state store ──────────────────────────────────────

# Marks a slot whose value is held unboxed in its `real` field. A singleton, so
# storing it allocates nothing.
struct _InReals end
const _IN_REALS = _InReals()

# One entry: its key, and its value either as stored or, for a value of the
# individual's own number type, unboxed in `real` with `_IN_REALS` in `val`.
# Such a value is boxed only when it is first read back as `Any`, and the box
# is kept in `val` so later reads reuse it, as a `Dict` does. Immutable, so a
# vector of them holds them inline, one allocation for the whole store.
struct _Slot{T <: Real}
    key::Symbol
    val::Any
    real::T
end

"""
    IndividualState{T}

What an [`Individual`](@ref)'s `state` holds: everything recorded about one
person beyond the core fields, keyed by `Symbol`. It is an
`AbstractDict{Symbol, Any}`, so it is read and written as a dictionary
(`ind.state[:onset_time]`, `get(ind.state, :age, missing)`,
`haskey(ind.state, :traced)`), and any `AbstractDict{Symbol, Any}` passed as
`state` when an `Individual` is built is copied into one.

Each value comes back as the type it was stored with, and keys are kept in
the order they were first set.
"""
struct IndividualState{T <: Real} <: AbstractDict{Symbol, Any}
    slots::Vector{_Slot{T}}
end

function IndividualState{T}() where {T <: Real}
    # Room for the few keys every individual gets; a store that interventions
    # add more to doubles as it grows.
    return IndividualState{T}(sizehint!(_Slot{T}[], 4))
end

function IndividualState{T}(d::AbstractDict) where {T <: Real}
    s = IndividualState{T}()
    for (k, v) in d
        s[k] = v
    end
    return s
end

IndividualState{T}(ps::Pair...) where {T <: Real} = IndividualState{T}(Dict{Symbol, Any}(ps...))

Base.convert(::Type{IndividualState{T}}, d::AbstractDict) where {T} = IndividualState{T}(d)
Base.convert(::Type{IndividualState{T}}, s::IndividualState{T}) where {T} = s

@inline function _slot(s::IndividualState, k::Symbol)
    slots = s.slots
    @inbounds for i in eachindex(slots)
        slots[i].key === k && return i
    end
    return 0
end
_slot(::IndividualState, k) = 0

@inline function _slot_value(s::IndividualState{T}, i::Int) where {T}
    slot = @inbounds s.slots[i]
    slot.val === _IN_REALS || return slot.val
    boxed = slot.real::Any
    @inbounds s.slots[i] = _Slot{T}(slot.key, boxed, slot.real)
    return boxed
end

@inline _make_slot(::Type{T}, k::Symbol, v) where {T} =
    v isa T ? _Slot{T}(k, _IN_REALS, v) : _Slot{T}(k, v, zero(T))

Base.length(s::IndividualState) = length(s.slots)
Base.haskey(s::IndividualState, k) = _slot(s, k) != 0

function Base.getindex(s::IndividualState, k)
    i = _slot(s, k)
    i == 0 && throw(KeyError(k))
    return _slot_value(s, i)
end

function Base.get(s::IndividualState, k, default)
    i = _slot(s, k)
    return i == 0 ? default : _slot_value(s, i)
end

function Base.get(f::Base.Callable, s::IndividualState, k)
    i = _slot(s, k)
    return i == 0 ? f() : _slot_value(s, i)
end

@inline function Base.setindex!(s::IndividualState{T}, v, k::Symbol) where {T}
    i = _slot(s, k)
    slot = _make_slot(T, k, v)
    if i == 0
        push!(s.slots, slot)
    else
        @inbounds s.slots[i] = slot
    end
    return s
end
Base.setindex!(s::IndividualState{T}, v, k) where {T} = setindex!(s, v, convert(Symbol, k)::Symbol)

function Base.get!(s::IndividualState, k, default)
    i = _slot(s, k)
    i == 0 || return _slot_value(s, i)
    s[k] = default
    return default
end

function Base.get!(f::Base.Callable, s::IndividualState, k)
    i = _slot(s, k)
    i == 0 || return _slot_value(s, i)
    v = f()
    s[k] = v
    return v
end

function Base.delete!(s::IndividualState, k)
    i = _slot(s, k)
    i == 0 || deleteat!(s.slots, i)
    return s
end

function Base.pop!(s::IndividualState, k)
    i = _slot(s, k)
    i == 0 && throw(KeyError(k))
    v = _slot_value(s, i)
    deleteat!(s.slots, i)
    return v
end

function Base.pop!(s::IndividualState, k, default)
    i = _slot(s, k)
    i == 0 && return default
    v = _slot_value(s, i)
    deleteat!(s.slots, i)
    return v
end

Base.empty!(s::IndividualState) = (empty!(s.slots); s)

function Base.iterate(s::IndividualState, i::Int = 1)
    i > length(s.slots) && return nothing
    return (Pair{Symbol, Any}((@inbounds s.slots[i]).key, _slot_value(s, i)), i + 1)
end

Base.copy(s::IndividualState{T}) where {T} = IndividualState{T}(copy(s.slots))
Base.empty(::IndividualState{T}) where {T} = IndividualState{T}()

# A time or other `T`-valued entry read without boxing it: the store's own
# lane when it holds one, a conversion otherwise, `default` (converted to `T`)
# when the key is absent.
@inline function _state_real(s::IndividualState{T}, k::Symbol, default::Real) where {T <: Real}
    i = _slot(s, k)
    i == 0 && return convert(T, default)::T
    slot = @inbounds s.slots[i]
    slot.val === _IN_REALS && return slot.real
    return convert(T, slot.val)::T
end
_state_real(s::AbstractDict, k, default::T) where {T} = convert(T, get(s, k, default))::T
