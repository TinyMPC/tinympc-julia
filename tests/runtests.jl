using Test
using TinyMPC

println("Running TinyMPC.jl Test Suite")
println("=" ^ 50)

@testset "TinyMPC.jl Tests" begin
    # Basic functionality tests
    println("\n🔧 Testing basic functionality...")
    include("test_basic.jl")
    
    # Cache functionality tests
    println("\n💾 Testing cache functionality...")
    include("test_cache.jl")
    
    # Settings functionality tests
    println("\n⚙️  Testing settings functionality...")
    include("test_settings.jl")
    
    # Sensitivity functionality tests
    println("\n🎯 Testing sensitivity functionality...")
    include("test_sensitivity.jl")
    
    # Code generation tests
    println("\n🔨 Testing code generation functionality...")
    include("test_codegen.jl")
end

println("\n" ^ 2)
println("✅ All tests completed successfully!")
println("TinyMPC.jl is working correctly.")
println("=" ^ 50) 