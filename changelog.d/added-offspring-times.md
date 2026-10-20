An offspring function can return an `OffspringTimes` instead of a count,
giving each contact's time from the window's `from` state directly and
skipping the kernel. This suits an offspring function that already knows
when each contact falls, such as one built day by day. `OffspringTimes`
has no per-type label, so it only works with a single-type model; a
multi-type model raises an error instead of assigning every contact to type
1.
