An offspring function can return an `OffspringTimes` instead of a count,
giving each contact's time from the window's `from` state directly and
skipping the kernel, for a model that already knows when its contacts fall,
such as one built day by day.
