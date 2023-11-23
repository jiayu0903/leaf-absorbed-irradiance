using CSV
using DataFrames
using DataFramesMeta
using ComponentArrays

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
using OrderedCollections

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

# Directory of the data
const dr = "./tomato-0908"
const format = dateformat"d/m/yyyy H:M:S p"

# Time points when the light was switched on
const tON = ("10:37:00 AM",
             "10:41:00 AM",
             "10:45:00 AM")

const dt = 5809.0
const date = "2023-09-08"

# Load sensor data
function Load_envir()
    df = CSV.File(dr * "/AirT2.csv") |> DataFrame

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

# Read timestamp from CSV thermograms
function getTS(tmp, t_init)
    ts = date * " " * split(tmp.Column1[1], " ")[3][5:end]
    dt = DateTime(ts, dateformat"yyyy-mm-dd H:M:S.s000")
    dtu = datetime2unix(dt)
    return  dtu + t_init
end

# Load thermograms
function Load_temps()
    # Build a mask
 img = load("Mask.bmp")
 s = size(img)
 target_height, target_width = 480, 640
 # Resize the image and fill the pixels using the interpolation method
 resized_img = imresize(img, (target_height, target_width))
 tmp = Gray.(resized_img)
 mask = tmp .> 0.0
 s = size(mask)
 maskr = reshape(mask, s[1] * s[2], 1)

    # List all CSV files
    files = readdir("CSV"; join = true)
    FN = [split(x[6], ".")[1] for x in split.(files, "_")]
    cd = OrderedDict(zip(parse.(Int64, FN), 1:length(files)))
    pos = Int64.(values(sort(cd)))

    # Read the time stamp
    tmp = CSV.File(files[pos[1]], skipto=3, limit=1, header=false)
    t = getTS(tmp, dt)
    
    # Read the first temperature matrix
    df = CSV.File(files[pos[1]], skipto=7, header=false) |> Tables.matrix

    ts = zeros(length(files))
    s = size(df)
    data = zeros(s[1] * s[2], length(files))

    ts[1] = t
    data[:,1] = reshape(df, s[1] * s[2], 1)

    for i in 2:length(files)
        tmp = CSV.File(files[pos[i]], skipto=3, limit=1, header=false)
        t = getTS(tmp, dt)
        
        # Read the temperature matrix
        df = CSV.File(files[pos[i]], skipto=7, header=false) |> Tables.matrix

        ts[i] = t
        data[:,i] = reshape(df, s[1] * s[2], 1).* maskr
    end

    ts .-= ts[1] + 54.0 # Time recorded before the sensor started 

    return TData(ts, (ts[1], ts[end]), s, data)
end

function LatentHeat(Tk)
    return 1.91846E6 * (Tk / (Tk - 33.91))^2
end

function SatVap(Tc)
    return 613.65 * exp(17.502 * Tc / (240.97 + Tc))
end

function Run()
    # Load datasets
    env = Load_envir()
    tps = Load_temps()

    # plot(tps.time, tps.mat[400,:], xlim=(0, 120))

    # Load the parameter values estimated by model_simple.jl
    # These parameters describe the relationship between temperature increase at t=30s
    # and the light intensity absorbed by the leaves
    a, b = load("ab_simple.jld2", "a", "b")

    # Select only the thermograms between t=0 and t=30s
    pos = findall(x -> 900.0 <= x <= 901.0 + 30.0, tps.time)

    # Exponential model to estimate Tleaf at t=30s because the data can be noisy
    # This way we use the whole kinetic to get a more robust estimate
    model(x, p) = @. p[1] - (p[2] - p[1]) * exp(-x / p[3])
    N = size(tps.mat)[1]
    Tl = zeros(N)

    # We apply the approach to each pixel of the thermogram at t=30s
    for i in 1:N
        x = tps.time[pos] .- 900.0
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
    Q_mod[Q_mod .> 50.0] .= 0.0
    hm = heatmap(Q_mod, yflip=true, c=:thermal)
    save("LightMap.png", hm)

    # Transpiration
    p = load("p_simple.jld2", "p")

    sp_Tair = CubicSpline(running_median(env.Tair, 3), env.time)
    sp_RH = CubicSpline(running_median(env.RH, 3), env.time)

    df = CSV.File(dr * "/ReflT.csv") |> DataFrame
    sp_Trefl = CubicSpline(running_median(df.Treflect, 3), df.Reltime)

    tt = 545.0
    Pa = env.Pa
    Tair = sp_Tair(tt)
    RH = sp_RH(tt)
    Trefl = sp_Trefl(tt)
    gbl = p.gbl / 1000.0

    pos = findall(x -> tt - 1.0 <= x <= tt + 1.0, tps.time)[1]
    Tleaf = @view tps.mat[:,pos]

    rho = Pa / (287.058 * (Tair + 273.0))
    ea = SatVap(Tair) * RH
    SH = 0.622 * ea / (Pa - ea)
    Cs = 1005.0 + 1820.0 * SH

    lambda = LatentHeat.(Tleaf .+ 273.0)
    LW = @. 2.0 * 0.97 * 5.6703e-8 * ((Trefl + 273.0)^4 - (Tleaf + 273.0)^4)
    C = @. 2.0 * rho * Cs * gbl * (Tleaf - Tair)
    E = @. (Q + LW - C - Tleaf) / lambda

    E_mod = reshape(E .* 1e3,  tps.size[1], tps.size[2])
    E_mod[E_mod .< 0.0] .= 0.0
    E_mod[E_mod .> 0.022] .= 0.0
    hm = heatmap(E_mod, yflip=true, c=:thermal)
    save("TrMap.png", hm)

    mean(E_mod[E_mod .> 0.0])
    # 计算300到454时间范围内的蒸腾速率的平均值
tt_start = 300.0
tt_end = 545.0

# 找到在指定时间范围内的索引
start_index = findfirst(x -> tt_start - 1.0 <= x <= tt_start + 1.0, tps.time)
end_index = findfirst(x -> tt_end - 1.0 <= x <= tt_end + 1.0, tps.time)

# 提取在指定时间范围内的蒸腾速率数据
E_range = E_mod[start_index:end_index, :]

# 计算平均值
average_et = mean(E_range[E_range .> 0.0])
println("300到454时间范围内的蒸腾速率平均值为: ", average_et)

end

Run()