`CapacityConstrained` with `carry_over` (the default) now keeps every period
within its cumulative allowance. A dose dated earlier than doses already
admitted counts against those later periods too, and is turned away if any of
them has no room left. Before, only the dose's own period was checked, so a
dose admitted after later ones could push those periods over their allowance.
