`CapacityConstrained` and `HeldCapacity` now share one internal supertype. It
holds the hook forwarding that each had written out separately, along with one
iterator over the people holding a capacity key. Behaviour, including the order
actions are admitted in, is unchanged.
