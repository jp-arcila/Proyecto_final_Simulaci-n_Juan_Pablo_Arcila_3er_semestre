#========================================================#
#||                      VARIABLES                      ||#
#========================================================#

#Generate points on the figures
points = circular_grid_cartesian(50)
points_sq = square_grid(30)
points_fl = flower_grid(60)

#Number of paths
N = 10^5

#define start points for Brownian paths
X, Y = map(p -> p[1], points), map(p -> p[2], points) #annilus
X_sq, Y_sq = map(p -> p[1], points_sq), map(p -> p[2], points_sq) #square
X_fl, Y_fl = map(p -> p[1], points_fl), map(p -> p[2], points_fl) #flower


#========================================================#
#||                      ANNILUS                       ||#
#========================================================#


#voltage with laplace equation
function Voltage_laplace_data(path::String = "./DATA/Voltage_annilus_laplace.csv")

    analytic_func(x, y) = sqrt(x^2 + y^2)^3 * cos(3 * atan(y, x)) #Analytic function for comparison

    Z = [multi_threads(x, y, N, boundary_func_voltage, exit_func_annilus, Brownian_motion_solver_laplace) for (x,y) in zip(X,Y)]
    W = analytic_func.(X, Y)

    d_set = dataset(X, Y, Z, W)
    rename!(d_set, :x => "x position", :y => "y position", :simulation_output => "Simulation Output (V)",
    :analytic_output => "Analytical Output (V)", :abs_error => "Absolure error (%)", :rel_error => "Relative error (%)")

    CSV.write(path, d_set)
end


#voltage with poisson equation
function Voltage_poisson_data(path::String = "./DATA/Voltage_annilus_poisson.csv", ϵ₀::Float64=8.8541878176e-12)

    r(x, y) = hypot(x, y)
    source_func(x, y) = r(x, y)/ϵ₀
    analytic_func(x, y) = r(x,y)^boundary_func_voltage(x,y) + (1 - r(x, y)^3)/(9*ϵ₀) #Analytic function for comparison

    #define datasets for dataframe
    Z = [multi_threads(x, y, N, boundary_func_voltage, exit_func_annilus, Brownian_motion_solver_poisson; source=source_func) for (x,y) in zip(X,Y)]
    W = analytic_func.(X, Y)

    d_set = dataset(X, Y, Z, W)
    rename!(d_set, :x => "x position", :y => "y position", :simulation_output => "Simulation Output (V)",
    :analytic_output => "Analytical Output (V)", :abs_error => "Absolute error (%)", :rel_error => "Relative error (%)")

    CSV.write(path, d_set)
end


#========================================================#
#||                       SQUARE                       ||#
#========================================================#


#Temperature of a square with easy solution
function Temp_data_easy(path::String="./DATA/Temp_data_easy.csv", X::Vector{Float64} = X_sq, Y::Vector{Float64} = Y_sq)
    sinh_pi = sinh(pi)  
    
    analytic_func(x, y) = sinh(π/2*(1-y))*sin(π/2*x)/sinh_pi
    Z = [multi_threads(x, y, N, boundary_square_easy, exit_func_square, Brownian_motion_solver_laplace) for (x, y) in zip(X, Y)] 
    W = analytic_func.(X, Y)

    d_set = dataset(X, Y, Z, W)
    rename!(d_set, :x => "x position", :y => "y position", :simulation_output => "Simulation Output (°C)",
    :analytic_output => "Analytical Output (°C)", :abs_error => "Absolute error (%)", :rel_error => "Relative error (%)")

    CSV.write(path, d_set)
end

#temperature of a square with hard solution
function Temp_data_square_hard(path::String="./DATA/Temp_data_hard.csv", X::Vector{Float64} = X_sq, Y::Vector{Float64} = Y_sq)
    Z = [multi_threads(x, y, N, boundary_square_hard, exit_func_square, Brownian_motion_solver_laplace) for (x, y) in zip(X, Y)] 
    W = fill(NaN, length(Z))
    d_set = dataset(X, Y, Z, W)
    rename!(d_set, :x => "x position", :y => "y position", :simulation_output => "Simulation Output (°C)",
    :analytic_output => "Analytical Output (°C)", :abs_error => "Absolute error (%)", :rel_error => "Relative error (%)")

    CSV.write(path, d_set)
end


#========================================================#
#||                       FLOWER                        ||#
#========================================================#


#temperature witin a flower figure r(θ) = cos(2*θ)
function Temp_data_flower(path::String="./DATA/Temp_data_flower.csv", X::Vector{Float64} = X_fl, Y::Vector{Float64} = Y_fl)
    analytic_func(x, y) = hypot(x, y)*cos(2*atan(y, x))
    Z = [multi_threads(x, y, 10^6, boundary_flower, exit_func_flower, Brownian_motion_solver_laplace) for (x, y) in zip(X, Y)] ./ 5 
    W = analytic_func.(X, Y)
    d_set = dataset(X, Y, Z, W)
    rename!(d_set, :x => "x position", :y => "y position", :simulation_output => "Simulation Output (°C)",
    :analytic_output => "Analytical Output (°C)", :abs_error => "Absolute error (%)", :rel_error => "Relative error (%)")

    CSV.write(path, d_set)
end
