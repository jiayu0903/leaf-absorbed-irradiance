using CSV
using DataFrames
using DataFramesMeta

using DataInterpolations
using Statistics
using Plots
using Measures
using FastRunningMedian

using DifferentialEquations
using SciMLSensitivity

using Distributions
using AdaptiveMCMC
using ComponentArrays
using StaticArrays

using FileIO
using Dates
using Tables

using Colors
using Images

# Define convenience log-transform for continuous univariate distributions
struct LogTransformedDistribution{Dist <: ContinuousUnivariateDistribution}
    d::Dist
end

import Distributions.logpdf
logpdf(d::LogTransformedDistribution, x) = logpdf(d.d, exp(x)) + x

import Base.log
log(d::ContinuousUnivariateDistribution) = LogTransformedDistribution(d)

# Environmental data (sensor)
struct EData
    time::Vector{Float64}
    tON::Vector{Float64}
    dt::Float64
    Tair::Vector{Float64}
    RH::Vector{Float64}
    Pa::Float64
    getTair::CubicSpline{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},true,Float64}
    getRH::CubicSpline{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},true,Float64}
end

# Temperature data (thermal camera)
struct TData
    time::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    Tl::Vector{Float64}
    Trfl::Vector{Float64}
    getTrfl::CubicSpline{Vector{Float64},Vector{Float64},Vector{Float64},Vector{Float64},true,Float64}
end

# Directory of the data
const dr = "./0908lettuce/lettuce-0908"
const date = "08/09/2023"
const format = dateformat"y-m-d H:M:S p"

# Time points when the light was switched on
const tON = ("05:20:00 PM",
             "05:24:00 PM",
             "05:28:00 PM")

# Load sensor data
function Load_envir()
    df = CSV.File(dr * "/AirT.csv") |> DataFrame

    t = datetime2unix.(DateTime.(string.(df.Date) .* " " .* df.Time, format))
    t .-= t[1] 
    t_on = datetime2unix.(DateTime.(string(df.Date[1]) * " " .* collect(tON), format))
    t_on .-= t_on[1]
    RH = df.Humidity ./ 100.0

    return EData(t,
                 t_on,
                 30.0,
                 df.Temperature,
                 RH,
                 101300.0,
                 CubicSpline(running_median(df.Temperature, 3), t),
                 CubicSpline(running_median(RH, 3), t))
end

# Load thermal camera data
function Load_temps()
    df = CSV.File(dr * "/LeafT.csv") |> DataFrame
    return TData(df.Time,
                 (df.Time[1], df.Time[end]),
                 df.Temperature,
                 df.Treflect,
                 CubicSpline(running_median(df.Treflect, 3), df.Time))
end

# Return the light intensity as a function of time 
function lcheck(t, env, Qabs)
    for i = eachindex(env.tON)
        if env.tON[i] <= t < (env.tON[i] + env.dt)
            return Qabs[i]
        end
    end 
    return 0.0
end

function LatentHeat(Tk)
    return 1.91846E6 * (Tk / (Tk - 33.91))^2
end

function SatVap(Tc)
    return 613.65 * exp(17.502 * Tc / (240.97 + Tc))
end

# Differential equations for photosynthesis and chlorophyll fluorescence
function NRGbal!(dy, y, p, t, env, tps)
    Tair = env.getTair(t)
    RH = env.getRH(t)
    Trefl = tps.getTrfl(t)
     
    rho = env.Pa / (287.058 * (Tair + 273.0))
    ea = SatVap(Tair) * RH
    SH = 0.622 * ea / (env.Pa - ea)
    Cs = 1005.0 + 1820.0 * SH

    gbl = p.gbl / 1000.0
    gsw = p.gsw / 1000.0
    Qabs = p.Qabs
    dy[1] = 2.0 * 0.97 * 5.6703e-8 * ((Trefl + 273.0)^4 - (y[1] + 273.0)^4)
    dy[1] += lcheck(t, env, Qabs)
    dy[1] -= 2.0 * rho * Cs * gbl * (y[1] - Tair)

    lambda = LatentHeat(y[1] + 273.0)
    vpd = SatVap(y[1]) - ea
    gtw_leaf = 1.0 / (1.0 / gsw + 0.92 / gbl)
    dy[1] -= lambda * 0.622 * rho / env.Pa * gtw_leaf * vpd

    dy[1] /= 1080.0
    return nothing
end

# Define the prior knowledge for bayesian inference
function log_priors(p)
    # Prior distributions
    lp = 0.0

    lp += sum(logpdf(Normal(0.0, 5.0), p.gbl))
    lp += sum(logpdf(Normal(0.0, 5.0), p.gsw))
    lp += sum(logpdf.(Normal(0.0, 100.0), p.Qabs))
    lp += sum(logpdf(Normal(0.0, 0.1), p.sT))

    return lp
end

iter::Int64 = 0

# Calculate the log posterior
function LP(p, prob, tps)
    global iter

    if any(p .< 0.0)
        iter += 1
        return -Inf
    end

    pos = 1:10:length(tps.time)
    
    # Solve the differential equation
    sol = solve(remake(prob, p=p), Tsit5(),
                saveat=view(tps.time, pos), maxiters=1e6, abstol=1e-6, reltol=1e-6,
                sensealg=InterpolatingAdjoint(checkpointing=true, autojacvec=ZygoteVJP()))
    
    tmp = Array(sol)::Matrix{Float64}

    if (size(tmp)[2] < length(tps.time) / 10.0)
        iter += 1
        return -Inf
    end

    lp = sum(logpdf.(Normal.(view(tps.Tl, pos), p.sT), tmp[1,:]))
    
    iter += 1
    println(string("Iter: ", iter, " - LP: ", lp, " - Priors: ", log_priors(p)))

    return lp
end

function bayes(env, tps)
    # Initial random parameter values
    gbl = 10.0 + 5.0 * (rand() * 2.0 - 1.0)
    gsw = 10.0 + 5.0 * (rand() * 2.0 - 1.0)
    Qabs = 100.0 .+ 50.0 * [(rand() * 2.0 - 1.0) for i in 1:length(env.tON)]
    sT = 0.2 + 0.1 * (rand() * 2.0 - 1.0)
    p = ComponentArray{Float64}(gbl=gbl, gsw=gsw, Qabs=Qabs, sT=sT)

    ode = (dy, y, p, t) -> NRGbal!(dy, y, p, t, env, tps)
    y0 = Float64[mean(tps.Tl[1:5])]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(ode, y0, tps.tspan, p)

    # Run the Metropolis-Hasting (RAM) algorithm
    model = x -> LP(x, prob, tps)
    out = adaptive_rwm(p, model, 100_000; log_pr=log_priors)

    p[:] = out.X[:,end] # Return the last p value

    return p
end

# Predict the leaf temperature kinetic
function pred(p, env, tps)
    ode = (dy, y, p, t) -> NRGbal!(dy, y, p, t, env, tps)
    y0 = Float64[mean(tps.Tl[1:5])]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(ode, y0, tps.tspan, p)

    sol = solve(remake(prob, p=p), Tsit5(),
                saveat=tps.time, maxiters=1e6, abstol=1e-6, reltol=1e-6,
                sensealg=InterpolatingAdjoint(checkpointing=true, autojacvec=ZygoteVJP()))
    
    return Array(sol)::Matrix{Float64}
end

# Plot model output
function plotSim(p, env, tps)
    tmp = pred(p, env, tps)

    p1 = plot(tps.time, tps.Tl, label="Tl obs.", linewidth=3,
              xlabel="Time (s)", ylabel="Leaf temperature (°C)",
              titlefontsize=18, guidefontsize=18,
              tickfontsize=16, legendfontsize=14)
    plot!(tps.time, view(tmp, 1, :), label="Tl mod.", linewidth=3)

    png("TempsKin.png")

    p1 |> display
end

# Plot the response curves at different light intensity
function deltaT(p, env, tps)
    tmp = pred(p, env, tps)

    p1 = plot(tps.time, tps.Tl, label="Tl obs.", xlim=[0.0, 31.0], legend=false)
    plot!(tps.time, view(tmp, 1, :), label="Tl mod.", color=:black)

    t = 0.0:0.1:30.0

    ode = (dy, y, p, t) -> NRGbal!(dy, y, p, t, env, tps)
    y0 = Float64[mean(tps.Tl[1:5])]
    prob = ODEProblem{true, SciMLBase.FullSpecialize}(ode, y0, (t[1], t[end]), p)

    pp = similar(p)
    pp[:] = p[:]
    Q = collect(0.0:10.0:450.0)
    Tl = similar(Q)

    for i = eachindex(Q)
        pp.Qabs[1] = Q[i]
        sol = solve(remake(prob, p=pp), Tsit5(),
                    saveat=t, maxiters=1e6, abstol=1e-6, reltol=1e-6,
                    sensealg=InterpolatingAdjoint(checkpointing=true, autojacvec=ZygoteVJP()))
    
        tmp = Array(sol)

        scatter!([30.0], [tmp[1, end]])
        plot!(t, view(tmp, 1, :), label=Q[i]) |> display
        Tl[i] = tmp[end]
    end

    p2 = scatter(Q, Tl, legend=false)
    a = Q \ (Tl .- Tl[1])
    plot!(Q, a .* Q .+ Tl[1]) |> display

    pg = plot(p1, p2, ncol=2)

    png("LightTemps.png")

    return a, Tl[1]
end

function Run()
    # Load datasets
    env = Load_envir()
    tps = Load_temps()

    p = bayes(env, tps)

    save("p_simple.jld2", "p", p)

    plotSim(p, env, tps)

    a, b = deltaT(p, env, tps)

    save("ab_simple.jld2", "a", a, "b", b)
end

Run()