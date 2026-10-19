`max_time` now stops each case from making further contacts once its own
infection reaches the limit, rather than ending the whole run as soon as any
case crosses it. A branch cut short this way is reported as not extinct, the
same as one stopped by `max_cases` or `max_generations`.
