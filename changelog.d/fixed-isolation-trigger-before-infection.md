An isolation-timed `TraceEligibility` (`OnIsolation()` and `TraceEveryone()`)
no longer times a case's forward trace from before the case was infected. A
person quarantined as a contact and then infected during that quarantine now
has their own trace timed from their infection at the earliest, rather than
from the start of the quarantine.
