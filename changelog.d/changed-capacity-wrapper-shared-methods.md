`CapacityConstrained` and `HeldCapacity` now share one internal supertype that
holds the hook forwarding they had each written out, and one iterator over the
people holding a capacity key. Behaviour, including the order actions are
admitted in, is unchanged.
