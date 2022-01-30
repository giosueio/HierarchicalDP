using Distributions
using Plots
using StatsBase
import Logging
Logging.disable_logging(Logging.Warn) # or e.g. Logging.Info

function stick_breaking(a, maxK)
    β = BigFloat[]
    prod = 1
    b = 1
    i=1
    while b*prod > 1e-10 && i < maxK
        b = rand(Beta(1,a))
        push!(β, b*prod)
        prod *= (1 - b)
        i += 1
    end
return β/sum(β) # Weights are normalized to ensure they sum to 1
end     

function DirichletProcess(gamma, H, maxK)
    β = stick_breaking(gamma, maxK)
    θ = rand(H, length(β))
return β, θ # indices and samples from H
end

function HierarchicalDP(a, DP)
    β = DP[1]
    π = Float64[]
    prod = 1
    for k in 1:(length(β)-1)
        p = rand(Beta(a*β[k], a*(1 - sum(β[1:k]))))
        push!(π,p*prod)
        prod *= (1-p)
    end
    ϕ = Categorical(π) 
    return ϕ # sampler of indices
end

function DP_sampler(DP,n)
    z_samples = rand(Categorical(DP[1]), n)
    theta_samples = DP[2][z_samples]
    
    a = countmap(theta_samples)
    b = [k for (k,d) in a]
    c = [d for (k,d) in a]
    c = c/sum(c)
    return theta_samples, b, c
end

function HDP_sampler(DP, G, n)
    theta_samples = DP[2][rand(G, n)] # use the G to sample first DP
    # x = sort!(sam) # Plot this for the CDF

    a = countmap(theta_samples)
    b = [k for (k,d) in a]
    c = [d for (k,d) in a]
    c = c/sum(c)
    return theta_samples, b, c
end

function DP_plot(DP_sample, H) # Takes output from a DP_sampler and the base measure H
    bar(DP_sample[2],DP_sample[3],yaxis = ([0,0.3], 0:0.05:1), xaxis = ("G0",(0:1), 0:0.1:1), legend=false)
    display(plot!(x->pdf(H,x)/20,line=(:dash)))
end

function HDP_plot(HDP_samples, H) # Takes a list of J HDP samples from HDP_sampler and the base measure H
    b = []
    for j in HDP_samples
        push!(b,bar(j[2],j[3],yaxis = ([0,0.6]), xaxis = ("G$j",(0:1))))
        plot!(x->pdf(H,x)/20,line=(:dash))
    end
    display(plot(b...,layout=(2,2), legend=false))
end