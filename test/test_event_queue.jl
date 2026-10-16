@testset "Event queue" begin
    Ev = EpiBranch._Event{Float64}

    # Pops at or after the clock, with many ties, past the switch to the radix
    # heap: the order must be exactly that of sorting every event queued so far.
    function queue_matches_sorting(seed; ties)
        rng = StableRNG(seed)
        q = EpiBranch._EventQueue{Float64}()
        waiting = Ev[]
        clock = 0.0
        for _ in 1:20_000
            for _ in 1:rand(rng, 0:3)
                t = ties && rand(rng) < 0.3 ? clock : clock + rand(rng) * 10
                e = EpiBranch._Event(t, rand(rng, 1:50), rand(rng, 1:5))
                EpiBranch._enqueue!(q, e)
                push!(waiting, e)
            end
            isempty(q) && continue
            e = EpiBranch._dequeue!(q)
            earliest, k = findmin(waiting)
            e == earliest || return false
            deleteat!(waiting, k)
            clock = e.time
        end
        return EpiBranch._radix(q)
    end
    @test all(s -> queue_matches_sorting(s; ties = true), 1:3)
    @test all(s -> queue_matches_sorting(s; ties = false), 1:3)

    @testset "Many events at one time pop by member" begin
        q = EpiBranch._EventQueue{Float64}()
        for k in 1:5000
            EpiBranch._enqueue!(q, EpiBranch._Event(0.0, k, 1))
        end
        EpiBranch._enqueue!(q, EpiBranch._Event(1.0, 1, 1))
        @test EpiBranch._radix(q)
        @test [EpiBranch._dequeue!(q).member for _ in 1:5000] == 1:5000
        @test EpiBranch._dequeue!(q).time == 1.0
    end

    @testset "Negative times order as times" begin
        q = EpiBranch._EventQueue{Float64}()
        EpiBranch._enqueue!(q, EpiBranch._Event(-10.0, 1, 0))
        EpiBranch._enqueue!(q, EpiBranch._Event(-1.0, 1, 0))
        @test EpiBranch._dequeue!(q).time == -10.0
        for i in 1:1100
            EpiBranch._enqueue!(q, EpiBranch._Event(Float64(i), 1, 0))
        end
        EpiBranch._enqueue!(q, EpiBranch._Event(-5.0, 2, 0))
        EpiBranch._enqueue!(q, EpiBranch._Event(-0.0, 2, 0))
        @test EpiBranch._radix(q)
        @test [EpiBranch._dequeue!(q).time for _ in 1:4] == [-5.0, -1.0, 0.0, 1.0]
    end

    @testset "An event before the last one taken is refused" begin
        q = EpiBranch._EventQueue{Float64}()
        for i in 1:2000
            EpiBranch._enqueue!(q, EpiBranch._Event(Float64(i), 1, 0))
        end
        @test EpiBranch._radix(q)
        EpiBranch._dequeue!(q)
        @test_throws ArgumentError EpiBranch._enqueue!(q, EpiBranch._Event(0.5, 1, 0))
    end

    @testset "Emptying keeps the order for what is queued next" begin
        q = EpiBranch._EventQueue{Float64}()
        for i in 1:2000
            EpiBranch._enqueue!(q, EpiBranch._Event(Float64(i), i, 0))
        end
        foreach(_ -> EpiBranch._dequeue!(q), 1:10)
        empty!(q)
        @test isempty(q)
        for t in (12.0, 11.0, 11.0)
            EpiBranch._enqueue!(q, EpiBranch._Event(t, Int(t), 0))
        end
        @test [EpiBranch._dequeue!(q).time for _ in 1:3] == [11.0, 11.0, 12.0]
    end

    @testset "Times other than Float64 stay on the binary heap" begin
        q = EpiBranch._EventQueue{Float32}()
        for i in 2000:-1:1
            EpiBranch._enqueue!(q, EpiBranch._Event(Float32(i), 1, 0))
        end
        @test !EpiBranch._radix(q)
        @test [EpiBranch._dequeue!(q).time for _ in 1:3] == Float32[1, 2, 3]
    end
end
