An `Individual`'s `state` no longer keeps up to three times the room it needs
once it grows past 16 keys on Julia 1.11 and later. It now doubles its capacity
when full, which cuts the memory held per individual under isolation and
contact tracing by about a third.
