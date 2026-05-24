for test_file in sort(readdir(@__DIR__; join=true))
    startswith(basename(test_file), "test_") || continue
    endswith(test_file, ".jl") || continue
    println("\n=== $(basename(test_file)) ===")
    include(test_file)
end
