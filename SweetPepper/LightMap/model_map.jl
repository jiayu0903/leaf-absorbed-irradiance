using CSV
using DataFrames
using DataFramesMeta

using DataInterpolations
using Statistics
using Plots
using Measures
using FastRunningMedian

using FileIO
using Dates
using Tables

using Colors
using Images

using LsqFit

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

# Thermograms
struct TData
    time::Vector{Float64}
    tspan::Tuple{Float64,Float64}
    size::Tuple{Int64,Int64}
    mat::Matrix{Float64}
end

# Time points when the light was switched on
const tON = ("08:42:00 PM",
             "08:46:00 PM",
             "08:50:00 PM",
             "08:54:00 PM",
             "08:58:00 PM")

const t0s = DateTime("21/12/2022 08:40:28 PM", dateformat"d/m/y H:M:S p")
const t0c = DateTime("21/12/2022 08:11:59 PM", dateformat"d/m/y H:M:S p")
const tst = DateTime("21/12/2022 08:42:00 PM", dateformat"d/m/y H:M:S p")
const date = "2022-12-21"

# Read timestamp from CSV thermograms
function getTS(tmp, t_init)
    ts = date * " " * split(tmp.Column1[1], " ")[3][5:end]
    dt = DateTime(ts, dateformat"yyyy-mm-dd H:M:S.s000")
    dtu = datetime2unix(dt)
    return  dtu + t_init
end

# Load thermograms
function Load_temps()
    # List all CSV files
    files = readdir("CSV"; join = true)
    FN = [split(x[6], ".")[1] for x in split.(files, "_")]
    ID = div.(parse.(Int64, FN) .+ 1, 16)
    pos = [findall(x -> x == i, ID)[1] for i in 1:length(files)]

    tsc = datetime2unix(t0s) - datetime2unix(t0c)

    # Read the time stamp
    tmp = CSV.File(files[pos[1]], skipto=3, limit=1, header=false)
    t = getTS(tmp, tsc)

    # Calculate the time difference between the start of the record
    # and the start of the first light pulse
    tdiff = datetime2unix(tst) - t
    
    # Read the first temperature matrix
    df = CSV.File(files[pos[1]], skipto=7, header=false) |> Tables.matrix

    ts = zeros(length(files))
    s = size(df)
    data = zeros(s[1] * s[2], length(files))

    ts[1] = t
    data[:,1] = reshape(df, s[1] * s[2], 1)

    for i in 2:length(files)
        tmp = CSV.File(files[pos[i]], skipto=3, limit=1, header=false)
        t = getTS(tmp, tsc)
        
        # Read the temperature matrix
        df = CSV.File(files[pos[i]], skipto=7, header=false) |> Tables.matrix

        ts[i] = t
        data[:,i] = reshape(df, s[1] * s[2], 1)
    end

    ts .-= ts[1] + tdiff
 
    pos = findall(x -> 0.0 <= x <= 1200.0 , ts)
    ts2 = @view ts[pos]
    data2 = view(data, :, pos)

    return TData(ts2, (ts2[1], ts2[end]), s, data2)
end

function Run()
    # Load datasets
    tps = Load_temps()

    # Load the parameter values estimated by model_simple.jl
    # These parameters describe the relationship between temperature increase at t=30s
    # and the light intensity absorbed by the leaves
    a, b = load("ab_simple.jld2", "a", "b")

    # Select only the thermograms between t=0 and t=30s
    pos = findall(x -> x <= 30.0, tps.time)

    # Exponential model to estimate Tleaf at t=30s because the data can be noisy
    # This way we use the whole kinetic to get a more robust estimate
    model(x, p) = @. p[1] - (p[2] - p[1]) * exp(-x / p[3])
    N = size(tps.mat)[1]
    Tl = zeros(N)

    # We apply the approach to each pixel of the thermogram at t=30s
    for i in 1:N
        x = tps.time[pos]
        y = tps.mat[i,pos]

        # Fit the model
        fit = curve_fit(model, x, y, [y[1], y[1] + 1, 10.0])

        # scatter(x, y)
        # plot!(x, model(x, fit.param)) |> display

        # Estimate Tleaf at t=30s using the model
        Tl[i] = model(30.0, fit.param)
    end

    # This code convert Tleaf at t=30s to a light intensity absorbed by the leaves
    Q = @. (Tl - b) / a
    Q[Q .< 0.0] .= 0.0

    # Rebuild an image from the matrix
    Q_mod = reshape(Q,  tps.size[1], tps.size[2])
    hm = heatmap(Q_mod, yflip=true, c=:thermal, clim=(0,250))
    save("LightMap.png", hm)
end

Run()