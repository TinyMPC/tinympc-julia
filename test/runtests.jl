println("Running TinyMPC.jl test suite…")

# Include individual test files
for tf in ["basic_test.jl", "simple_demo.jl", "run_examples_test.jl"]
    include(tf)
end

println("All tests finished.") 