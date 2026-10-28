`HazardKernel(hazard; upper = Inf)` builds a contact-interval distribution
directly from an instantaneous hazard rather than a named family, for use as
an `Infectiousness` `kernel` that follows something computed per case, such
as a within-host trajectory drawn once at infection and read through a link
turning it into a transmission rate.
