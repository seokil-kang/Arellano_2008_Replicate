#= This code replicates Arellano, AER 2008
"Default Risk and Income Fluctuations in Emerging Economies"
Code written by Seokil Kang, July 2019
=#

clearconsole()

# call required packages
using LinearAlgebra, QuantEcon, Optim, Plots, DataFrames, CSV, CPUTime

# checkpoint for computation running-time
CPUtic()

# parameters and calibration
r = .017        # risk-free interest rate
σ = 2           # risk aversion
ρ = .945        # income process AR(1) persistence
η = .025        # income process AR(1) volatility
β = .953        # discount factor
θ = .282        # probability of reentry
y_hat = .969    # output costs

# computation setup
n_B = 200       # number of asset grid
n_y = 21        # number of income state

# asset grid
B_min = -.35    # minimum of asset grid
B_max = .15     # maximum of asset grid

# construct asset grid
# use collect to make B float[64]
B = collect(range(B_min, B_max, length = n_B))

# insert zero point in asset grid
if sum(findall(x-> x==0, B)) < 1
    B = sort([B;0])
    # redefine asset grid number
    n_B = length(B)
end

# keep the zero asset index
zero_asset = searchsortedfirst(B,0)

# income grid: Markov chain
# create Markov Chain structure
mc = tauchen(n_y, ρ, η)

# transition probability
# f_ij = prob of trans from state i to j
# sum of each row is one.
f = mc.p

# income y_grid
y = exp.(mc.state_values)

# initial bond price schedule
q_0 = repeat([1/(1+r)], n_B, n_y)

# value function basket
V_c_old = zeros(n_B, n_y)   # value of repayment old
V_d_old = zeros(n_y, 1)     # value of default old
V_o_old = zeros(n_B, n_y)   # value function old
V_c_new = zeros(n_B, n_y)   # value of repayment new
V_d_new = zeros(n_y, 1)     # value of default new
V_o_new = zeros(n_B, n_y)   # value function new

# policy function basket
Bond = zeros(n_B, n_y)          # policy: B'
Bond_index = zeros(n_B, n_y)    # policy: B' index

# price basket
q = zeros(n_B, n_y)             # bond price schedule
δ = zeros(n_B, n_y)             # default probability

# Solve model via VFI

# price convergence setup
q_err = 1           # initial convergence error
q_tol = 1e-3        # convergence tolerance level
q_iter = 0          # iteration setup

while q_err > q_tol
    # some variables require global declaration in loop...
    global q_err, q_tol, q_iter, VFI_iter, VFI_tol, VFI_err, V_o_new, q, q_0, δ
    # update iteration number
    q_iter += 1
    # report convergence status
    println("price updating count = $(q_iter) with error = $(round(q_err; digits = 4))")

    # VFI setup
    VFI_err = 1         # initial convergence error
    VFI_tol = 1e-2      # convergence tolerance level
    VFI_iter = 0        # iteration setup

    while VFI_err > VFI_tol
        # some variables require global declaration in loop...
        global VFI_err, VFI_tol, VFI_iter, V_o_new
        # update iteration number
        VFI_iter += 1
        # report iteration status
        if VFI_iter % 25 == 0
            println("VFI count = $(VFI_iter) with error = $(round(VFI_err; digits = 4))")
        end

        # update values
        V_c_old = V_c_new
        V_d_old = V_d_new
        V_o_old = V_o_new

        # loop for state: income y
        for j in 1:n_y
            # loop for state: asset holdings B
            for i in 1:n_B
                # compute consumption with resource constraint
                # keep consumption greater than zero through max function
                c = max.(y[j].+B[i].-q_0[:,j].*B, 1e-10)
                # compute value of repayment for all asset holdings B
                v = c.^(1-σ)./(1-σ) + β.*V_o_old*f[j,:]
                # pick the maximum value
                vmax,  = findmax(v)
                # record the maximizing index
                # if the maximizer is not unique, pick the last one
                # which locates on the reasonable side of Laffer curve
                indx = searchsortedlast(v,vmax)
                # ensure the index is the right maximizer
                if v[indx] != vmax
                    vmax, indx = findmax(v)
                end
                # record the solutions in each baskets
                V_c_new[i,j] = vmax
                Bond_index[i,j] = indx
                Bond[i,j] = B[indx]
            end
            # compute output at default
            y_def = min.(y_hat*mean(y), y[j])
            # instant utility at default
            vd1 = y_def^(1-σ)/(1-σ)
            # present value of default in future
            vd2 =β.* (θ.*V_o_old[zero_asset,:]' .+ (1-θ).*V_d_old')*f[j,:]
            # Julia does not regard vd2 as a scalar...
            V_d_new[j] = vd1 + vd2[1]
        end
        # compute value function
        V_o_new = max.(V_c_new, repeat(V_d_new',n_B,1))
        # measure the error of iteration
        VFI_err = norm(V_o_new-V_o_old)
    end

    # default probability
    δ = (V_d_new' .> V_c_new) * f'

    # equilibrium bond pricing
    q = (1 .- δ) ./ (1+r)

    # updating new equilibrium bond pricing
    q_err = norm(q-q_0)
    α = .5
    q_0 = α .* q_0 + (1-α).* q
end

# compute some equilibrium state_values
# total resource borrowed
QB = q.*Bond

# consumption
c = repeat(y',n_B,1) + repeat(B,1,n_y) - QB

println("Computation is complete")

# checkpoint for computation running-time
CPUtoc()

# save results
# grids need modifying for dataframing
BB = zeros(n_B, 1)
for i in 1:n_B
    BB[i] = B[i]
end
Bgrid = DataFrame(BB)
yy = zeros(n_y, 1)
for i in 1:n_y
    yy[i] = y[i]
end

# DataFrame function for output record
y_grid = DataFrame(yy)
transP = DataFrame(f)
default_prob = DataFrame(δ)
bondprice = DataFrame(q)
policy_B = DataFrame(Bond)
policy_c = DataFrame(c)
valfn = DataFrame(V_o_new)

# write csv files for variables
CSV.write("result_Bgrid.csv", Bgrid)
CSV.write("result_ygrid.csv", y_grid)
CSV.write("result_transP.csv", transP)
CSV.write("result_default_prob.csv", default_prob)
CSV.write("result_bondprice.csv", bondprice)
CSV.write("result_policy_B.csv", policy_B)
CSV.write("result_policy_c.csv", policy_c)
CSV.write("result_valfn.csv", valfn)

# report results
gr()   # looks like plotting on the pane
i_low = searchsortedlast(y,mean(y)*.95)
i_high = searchsortedlast(y,mean(y)*1.05)

# bond price schedule plot
Q = [q[:,i_low] q[:,i_high]]
p1 = plot(B, Q, lw=2, lab="")
title!("Bond pricing schedule")
xlabel!("asset")
xlims!(-.35, .0)

# equilibrium interest rate plot
R = [min.((1 ./q[:,i_low]).^4 .- 1, .1) min.((1 ./q[:,i_high]).^4 .- 1, .125)]
p2 = plot(B, R, lw=2, label = ["y_low" "y_high"], legend = :topright)
title!("Equilibrium interest rate")
xlabel!("asset")
xlims!(-.08, .03)

# savings function plot
BOND = [Bond[:,i_low] Bond[:,i_high]]
p3 = plot(B, BOND, lw=2, lab="")
title!("Savings function")
xlabel!("asset")
xlims!(-.3, .175)

# value function plot
V = [V_o_new[:,i_low] V_o_new[:,i_high]]
p4 = plot(B, V, lw=2, lab="")
title!("Value function")
xlabel!("asset")
xlims!(-.3, .175)

# make subplots
plot(p1, p2, p3, p4, layout = 4)
