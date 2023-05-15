# dy/dx = sin(x/n)/abs(y) - abs(x/y)
using Plots;

n = 1
dx = 0.01
iters = 500
init_pt = (0., -2.)

dydx(pt) = sin(pt[1]/n)/abs(pt[2]) - abs(pt[1] / pt[2])

f(pt) = (pt[1] + dx, pt[2]+ dydx(pt) * dx)


pt = init_pt
pts = [pt]
for i in 1:iters
    println(pt)
    global pt = f(pt)
    push!(pts, pt)
end



pt = init_pt
dx = -dx
for i in 1:iters
    println(pt)
    global pt = f(pt)
    pushfirst!(pts, pt)
end



println(pts)
plot!(pts, show=true)
readline()
