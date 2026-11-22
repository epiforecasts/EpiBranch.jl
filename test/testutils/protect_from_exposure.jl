using EpiBranch

# Protects every contact exposed from `from_time` on, reading the exposure from
# the contact's `infection_time`, which holds the proposed time while the
# risks are resolved.
struct ProtectFromExposure <: EpiBranch.AbstractIntervention
    from_time::Float64
end
function EpiBranch.competing_risk(p::ProtectFromExposure, parent, contact, state)
    return Risk(
        block_probability = (rng, parent, contact, state) -> contact.infection_time >=
            p.from_time ? 1.0 : 0.0
    )
end
