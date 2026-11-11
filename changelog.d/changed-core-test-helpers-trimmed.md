The core test suite no longer keeps copies of the satellite test helpers it never calls. The two satellite packages keep their own copies, each pointing at the other.
