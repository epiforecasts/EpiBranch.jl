`HouseholdProcess` now throws an `ArgumentError` for a negative or `NaN` `obs_end`, as `NetworkProcess` and `RoutedNetwork` already did. The three constructors share one check, `_check_obs_end`.
