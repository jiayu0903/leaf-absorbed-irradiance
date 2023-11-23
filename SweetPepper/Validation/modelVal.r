library(deSolve)
library(ggplot2)
library(cowplot)
library(cmdstanr)

theme_set(theme_cowplot())

# Provide the light intensity as a function of time
lcheck <- function(t, ll, p) {
  n <- length(p)
  
  for (i in 1:n) {
    if (t > ll$tON[i] & t < (ll$tON[i] + ll$dt[i]))
      return(p[i])
  }
  
  return(0)
}

# Calculate the dew point vapour pressure
SatVap <- function(Tc) {
  613.65 * exp(17.502 * Tc / (240.97 + Tc)) # Pa
}

LatentHeat <- function(Tk) {
  1.91846E6 * (Tk / (Tk - 33.91))^2 # J Kg-1
}
# Differential equations for energy balance
derivs <- function(t, y, p, ll) {
  Tair <- predict(ll$sspl_A, t)$y
  RH <- predict(ll$sspl_R, t)$y / 100
  Treflect <- predict(ll$sspl_E,t)$y
  Pa <- 101300
  
  rho <- Pa / (287.058 * (Tair + 273))
  ea <- SatVap(Tair) * RH
  SH <- 0.622 * ea / (Pa - ea)
  Cs <- 1005 + 1820 * SH
  
  SW <- lcheck(t, ll, tail(p, ll$sl))
  LW <- 2 * 0.97 * 5.6703E-8 * ((Treflect + 273)^4 - (y[1] + 273)^4) # Approx.
  R <- SW + LW
  
  C <- 2 * rho * Cs * p[1] *(y[1] - Tair)
  lambda = LatentHeat(y[1] + 273)
  vpd = SatVap(y[1]) - ea;
  gtw_leaf = 1. / (1. / p[2] + 0.92 / p[1]);
  lE = lambda * 0.622 * rho / Pa * gtw_leaf * vpd;
  
  
  
  dydt1 <- (R - C - lE) / 800
  return(list(dydt1))
}

# Directory of the data
dr <- "30cmrandom/pepper-1025-30cm/"
date <- "2022-10-25"
time <- "21:05:00"

# Start time of the temperature records
t0 <- as.POSIXct(paste(date, time), format="%Y-%m-%d %H:%M:%OS")

# Time points when the light was switched on
tON <- c("21:05:00",
         "21:09:00",
         "21:13:00",
         "21:17:00",
         "21:21:00")

# Observed vs. estimated light intensity (different units)
PAR <- c(128.7322691,
         111.271802,
         186.2555652,
         144.4270982,
         168.1329206)

# Read and process air temperature data
dA <- read.csv(paste0(dr, "AirT.csv"))
dA$time <- as.POSIXct(paste(date, dA$Time), format="%Y-%m-%d %I:%M:%OS %p")
ggplot(dA, aes(x=time, y=Temperature)) + geom_point()
ggplot(dA, aes(x=time, y=Humidity)) + geom_point()

# Read and process leaf temperature data
dL <- read.csv(paste0(dr, "LeafT.csv"))
names(dL) <- c("Date", "time", "Temperature", "Tair", "Treflect")

# Parse date string
dL$ts <- as.POSIXct(paste(date, dL$Date), format="%Y-%m-%d %I:%M:%OS %p")
ggplot(dL, aes(x=ts, y=Temperature)) + geom_point()
ggplot(dL, aes(x=ts, y=Tair)) + geom_point()

# Find initial time common to both data sets
dA$time <- as.numeric(dA$time - t0)
dL$time <- dL$time + as.numeric(dL$ts[1] - t0)

dA <- subset(dA, time >= 0)
dL <- subset(dL, time >= 0)

# Use splines to extract data at common time points
spl_A <- splinefun(dA$time, dA$Temperature)
spl_R <- splinefun(dA$time, dA$Humidity)
spl_L <- splinefun(dL$time, dL$Temperature)
spl_A2 <- splinefun(dL$time, dL$Tair)
spl_E <- splinefun(dL$time, dL$Treflect)

fitt <- smooth.spline(dA$time, dA$Humidity, cv=TRUE)
fitt2 <- smooth.spline(dA$time, dA$Temperature, cv=TRUE)
fitt3 <- smooth.spline(dL$time, dL$Tair, cv=TRUE)
fitt4 <- smooth.spline(dL$time, dL$Treflect, cv=TRUE)

fitt$df
fitt2$df
fitt3$df
fitt4$df



# Remove high frequency fluctuations
sspl_R <- smooth.spline(dA$time, dA$Humidity, df=65)
sspl_A <- smooth.spline(dA$time, dA$Temperature, df=38)
sspl_A2 <- smooth.spline(dL$time, dL$Tair, df=206)
sspl_E <- smooth.spline(dL$time, dL$Treflect, df=205)


# Check spline function
plot(Humidity~time, dA, pch=16)
points(predict(sspl_R, time)$y~time, dA, lty=2, lwd=2, col="red")

plot(Temperature~time, dA, pch=16)
lines(predict(sspl_A, time)$y~time, dL, lty=2, lwd=2, col="red")

plot(Tair~time, dL, pch=16)
lines(predict(sspl_A2, time)$y~time, dL, lty=2, lwd=2, col="red")

# New time points
time <- subset(dL, time < max(dA$time))$time

# Merge the data sets
d <- data.frame(time=time, Ta=predict(sspl_A, time)$y,
                Ta2=predict(sspl_A2, time)$y,
                Tl=spl_L(time),
                Treflect=predict(sspl_E, time)$y,
                RH=predict(sspl_R, time)$y)

# Correct for offset between Rotronic sensor and thermal camera temperature
#d$Tl <- d$Tl - (tail(d$Tl) - tail(d$Ta))

ggplot(d, aes(x=time)) + geom_point(aes(y=Ta)) +
  geom_point(aes(y=Ta2), col="blue") +
  geom_point(aes(y=Tl), col="red") + geom_point(aes(y=Treflect), col="green") + ylab("Temperature")
ggplot(d, aes(x=time, y=Tl - Ta)) + geom_point() + ylab("Tl - Ta")


# Function to be passed to the ODE solver
l <- NULL
l$sspl_A <- sspl_A
l$sspl_R <- sspl_R
l$sspl_E <- sspl_E

# Time light was turned ON
l$tON <- as.POSIXct(paste(date, tON), format="%Y-%m-%d %H:%M:%S")
l$tON <- as.numeric(l$tON - t0)

l$ts <- diff(c(0, l$tON, max(d$time)))
l$dt <- rep(30, length(l$tON))

l$st <- length(l$tON)+2
l$sl <- length(l$tON)

# Read calibrated parameter values
fit <- readRDS("30cmordered/fit.RDS")
fs1 <- fit$summary("gbl")
fs2 <- fit$summary("gsw")
fs2 <- fs2
li_Cal <- read.csv("30cmordered/Light.csv", header=F)

lm_out <- lm(li_Cal[,2]~li_Cal[,1]-1)
plot(li_Cal, xlim=c(0,250), ylim=c(0,250), pch=16, xlab="PARobs", ylab="PARmod")
abline(lm_out, lty=2)

p <- c(fs1$mean, fs2$mean, coef(lm_out)[1]*PAR)
p


# ODE solver
ode.out <- lsodes(d$Tl[1], d$time, derivs, p, ll=l)

# Display predicted temperature kinetic
png("Kinetic.png", width=1600, height=1200, res=250)
par(mar=c(5,5,2,2), cex=1)
Tmin <- min(d$Tl, ode.out[,2])
Tmax <- max(d$Tl, ode.out[,2])
plot(Tl~time, d, pch=16, xlab="Time (s)", ylab="Tleaf", ylim=c(Tmin, Tmax))
points(ode.out[,2] ~ ode.out[,1], col="red", type="l", lwd=2, lty=2)
dev.off()


write.table(ode.out, "Temperature.csv", row.names=F, col.names=F, sep=",")
write.table(d$Tl, "Temperature-Obs.csv", row.names=F, col.names=F, sep=",")

d <- ode.out[,2]
print(d)
