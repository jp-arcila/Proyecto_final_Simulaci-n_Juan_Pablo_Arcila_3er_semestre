#========================================================#
#||                 EXIT FUNCTIONS                     ||#
#========================================================#

#Exit function for the annilus x² + y² = 1
function exit_func_annilus(x, y)
    return x^2 + y^2 >= 1
end

#Exit function for a square of side l = 2 centered at the origin
function exit_func_square(x, y)
    return x>=1 || y>=1 || y<=-1 || x <=-1
end

#Exit function for the flower r(θ) = cos(2θ)
function exit_func_flower(x, y)
    r = hypot(x, y)
    θ = atan(y, x)
    return r>=abs(cos(2θ)) || r == 0  # Límite de pétalos
end


#========================================================#
#||                 BOUNDARY CONDITIONS                ||#
#========================================================#


function boundary_func_voltage(x, y)
    return cos(3*atan(y, x))
end

function boundary_square_easy(x, y)
    if x >= 1 || x<= -1 || y >=1
        return 0.0
    elseif y<=-1
        return sin(π/2 * x)
    end
end

function boundary_square_hard(x, y)
    if x >= 1
        return 10*y
    elseif x <= -1
        return 30*y
    elseif y >= 1
        return 20*x
    elseif y <= -1
        return 40*x
    end
end

function boundary_flower(x, y)
    r = hypot(x, y)
    θ = atan(y, x)
    
    if r ≈ 0.0  # Inner circle (Dirichlet)
        return 0.0
    elseif exit_func_flower(x, y)
        return 5.0 * cos(2θ)
    end
end

#========================================================#
#||                MONTE CARLO FUNCTIONS               ||#
#========================================================#

#multi threads implementation for faster computing 
function multi_threads(x::Float64, y::Float64, N::Int, boundary_func::Function, exit_func::Function, solver::Function; source::Union{Function, Nothing}=nothing)  
    nthreads = Threads.nthreads()
    results = zeros(nthreads)
    chunksize = cld(N, nthreads)
    Threads.@threads for i in 1:nthreads
        results[i] = if isnothing(source) 
            solver(x, y, chunksize, boundary_func, exit_func)
        else 
            solver(x, y, chunksize, boundary_func, exit_func, source)
        end
    end

    return mean(results)
end

#Monte Carlo solver for Laplace equations ∇²u(x) = 0
function Brownian_motion_positions_laplace(x0::Float64, y0::Float64, N::Int, exit_func::Function, δ=0.001)
    full_pos = []
    σ = sqrt(δ)
    for _ in 1:N
        pos = []
        x, y = x0, y0
        while true
            push!(pos, [x, y])
            
            # Generate Gaussian increments for Brownian motion
            dx = σ * randn()
            dy = σ * randn()
            x, y = x + dx, y + dy
            
            if exit_func(x, y)
                push!(pos, [x, y])
                break
            end
        end
        push!(full_pos, pos)
    end
    
    return full_pos
end

#Monte Carlo solver for Poissont ecuations ∇²u(x) = g(x)
function Brownian_motion_solver_poisson(x0::Float64, y0::Float64, N::Int, frontier_cond::Function, exit_func::Function, source_func::Function, δ::Float64=0.001)

    # Initialize the Brownian motion solver
    ϕ = 0.0
    σ = sqrt(δ)
    δ_2 = δ^2
    for _ in 1:N
        x, y = x0, y0
        β = 0.0
        while true
            # Generate Gaussian increments for Brownian motion
            dx = σ * randn()
            dy = σ * randn()
            x_new, y_new = x + dx, y + dy
            β += source_func(x_new, y_new) * δ_2
            
            if exit_func(x_new, y_new)
                # Project exit point to the unit circle for accuracy
                r = hypot(x_new, y_new)
                x_exit = x_new / r
                y_exit = y_new / r
                ϕ += frontier_cond(x_exit, y_exit)
                break
            else
                x, y = x_new, y_new
            end
            ϕ += β
        end
    end
    return ϕ/N  # Average over all paths  
end

#Function that saves the Monte Carlo walks for any figure
function Brownian_motion_solver_laplace(x0::Float64, y0::Float64, N::Int, frontier_cond::Function, exit_func::Function, δ::Float64=0.001)
    ϕ = 0.0
    σ = sqrt(δ)
    for _ in 1:N
        x, y = x0, y0
        while true
            dx = σ * randn()
            dy = σ * randn()
            x_new, y_new = x + dx, y + dy
            
            if exit_func(x_new, y_new)
                boundary_value = frontier_cond(x_new, y_new)
                if !(boundary_value isa Number)
                    error("Boundary condition returned non-numeric value: $boundary_value")
                end
                ϕ += boundary_value
                break
            else
                x, y = x_new, y_new
            end
        end
    end
    return ϕ / N
end

#========================================================#
#||                   GRID FUNCTIONS                   ||#
#========================================================#

function circular_grid_cartesian(n::Int, radius::Float64=1.0)
    x = range(-radius, radius, length=n)
    y = range(-radius, radius, length=n)
    points = Tuple{Float64, Float64}[]
    for xi in x, yi in y
        if hypot(xi, yi) < radius
            push!(points, (xi, yi))
        end
    end
    return points
end

function square_grid(n::Int, radius::Float64=1.0)
    x = range(-radius, radius, length = n)
    y = copy(x)
    points=[]
    for xi in x, yi in y
        if abs(xi)<radius && abs(yi)<radius
            push!(points, (xi, yi))
        end
    end
    return points
end

function flower_grid(n::Int, radius::Float64=1.0)
    x = collect(range(-radius, radius, length=n))
    y = copy(x)
    points = []
    for xi in x, yi in y 
        if exit_func_flower(xi, yi) == false
            push!(points, (xi, yi))
        end
    end
    return points
end


#========================================================#
#||          DATA ORGANIZATION FUNCTIONS               ||#
#========================================================#

#Error calculator
function abs_rel_error(simulation_output, analytic_output)
    absolute_error = [abs(s - a) for (s, a) in zip(analytic_output, simulation_output)]
    relative_error = [100*abs.((s - a)/(a)) for (s, a) in zip(analytic_output, simulation_output)]
    return [round.(absolute_error, digits=2), round.(relative_error, digits=2)]
end 

#Dataframe for position, outputs and errors
function dataset(X::Vector{Float64}, Y::Vector{Float64}, simulation_output::Vector{Float64}, analytic_output::Vector{Float64})
    errors = abs_rel_error(simulation_output, analytic_output)
    df = DataFrame(x=X, y=Y, simulation_output=simulation_output, analytic_output=analytic_output, abs_error=errors[1], rel_error=errors[2])
    return df  
end



