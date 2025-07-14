# courtesy of Kevin Tracy
import MeshCat as mc
using ColorTypes
using GeometryBasics: HyperRectangle, Cylinder, Vec, Point, Mesh
using CoordinateTransformations
using Rotations

function rotx(θ)
    s, c = sincos(θ)
    return [1 0 0; 0 c -s; 0 s c]
end
function create_cartpole!(vis)
    mc.setobject!(vis[:cart], mc.HyperRectangle(mc.Vec(-.25,-1.0,-.15), mc.Vec(0.5,2,0.3)))
    mc.setobject!(vis[:pole], mc.Cylinder(mc.Point(0,0,-.75), mc.Point(0,0,.75), 0.05))
    mc.setobject!(vis[:a], mc.HyperSphere(mc.Point(0,0,0.0),0.1))
    mc.setobject!(vis[:b], mc.HyperSphere(mc.Point(0,0,0.0),0.1))
end
function update_cartpole_transform!(vis,x)
    pole_o = 0.3
    px = x[1]
    θ = x[3]
    mc.settransform!(vis[:cart], mc.Translation([0,px,0.0]))
    p1 = [pole_o,px,0]
    p2 = p1 + 1.5*[0, sin(θ), -cos(θ)]
    mc.settransform!(vis[:a], mc.Translation(p1))
    mc.settransform!(vis[:b], mc.Translation(p2))
    mc.settransform!(vis[:pole], mc.Translation(0.5*(p1 + p2)) ∘ mc.LinearMap(rotx(θ))) 
end

function animate_cartpole(X, dt)
    print(length(X))
    vis = mc.Visualizer()
    create_cartpole!(vis)
    anim = mc.Animation(floor(Int,1/dt))
    for k = 1:length(X)
        mc.atframe(anim, k) do
            update_cartpole_transform!(vis,X[k])
        end
    end
    mc.setanimation!(vis, anim)
    return mc.render(vis)
end

function mat_from_vec(X)
    # convert a vector of vectors to a matrix 
    Xm = hcat(X...)
    return Xm 
end

function visualize_quad_state(X)
    # visualize the state history of the quadrotor 
    X_m = mat_from_vec(X)
    display(plot(X_m[1:7,:]',label=["x" "y" "z" "qw" "qx" "qy" "qz"],
    linestyle=[:solid :solid :solid :dash :dash :dash :dash], linewidth=[2 2 2 2 2 2 2],
                 title="State History", xlabel="time (s)", ylabel="x"))
end

function visualize_quad_xy(Xreal, Xref=nothing)
    # visualize the xy position of the quadrotor
    if Xref != nothing
        X_m = mat_from_vec(Xref)
        plot(X_m[2,:],X_m[1,:],label="ref",
        linestyle=:solid, linewidth=2,
                     title="State History", xlabel="y", ylabel="x")
        X_m = mat_from_vec(Xreal)   
        display(plot!(X_m[2,:],X_m[1,:],label="real", linestyle=:dash, linewidth=2,
                    title="State History", xlabel="y", ylabel="x", aspect_ratio=:equal))
    else
        X_m = mat_from_vec(Xreal)   
        display(plot(X_m[2,:],X_m[1,:],label="real", linestyle=:dash, linewidth=2,
                    title="State History", xlabel="y", ylabel="x", aspect_ratio=:equal))
    end
end