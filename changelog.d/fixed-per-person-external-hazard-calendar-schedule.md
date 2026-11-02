`external_hazard` accepts a calendar schedule such as `Steps` per person, as a
vector or a function of the person's number, rather than only one shared by
everyone: a model where each cluster has its own calendar of community
incidence no longer has to write its own piecewise-constant distribution for
every cluster. A function returning a schedule is also checked when the
process is built, rather than only once the model is simulated.
