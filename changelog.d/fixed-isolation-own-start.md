`Scheduled` and `CapacityConstrained` use isolation's own start time when
judging an `Isolation`, so an existing quarantine does not shift its schedule
or charge it to the wrong capacity period.
