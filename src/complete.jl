"""
Complete Trading

Symbols in comments for set and parameter structs indicate the symbols used in the paper.

"""

using JSON, JuMP, CPLEX, DataStructures

struct sets
    GENS # g
    TIME_BLOCKS # t
    FUEL_PRICES # f
    PROFILES # r
    SHIFTERS # s
end

struct parameters
    VOLL # B
    DURATION # L
    LOAD_FIX # D_t^{fix}
    LOAD_FLEX # F_t^{res}
    AVAILABILITY # A_{grt}
    MARG_COST # C_{fg}^{EN}
    LOAD_SHIFT_NEG # D_f^-
    LOAD_SHIFT_POS # D_f^+
    α
    β
    SCEN_PROB # p_{frs}
    GEN_INV_COST # C_g^{INV}
end

mutable struct evaluation
    D_flex
    α
    β
    CAPACITY
    SURPLUS
    Objective
    Unserved_energy
    AVG_spot_price
    spot_price_std
    VAR
    CVAR
end

function init_evaluation(param::Dict)
    D_flex = 0.0
    α = 0.0
    β = 0.0
    CAPACITY = Dict(g => 0.0 for g in param["GENS"])
    SURPLUS = 0
    Objective = 0
    Unserved_energy = 0
    AVG_spot_price = 0.0
    spot_price_std = 0.0
    VAR = 0
    CVAR = 0
    return evaluation(D_flex, α, β, CAPACITY, SURPLUS, Objective, Unserved_energy, AVG_spot_price, spot_price_std, VAR, CVAR)
end

function init_sets(param::Dict)
    GENS = param["GENS"]
    TIME_BLOCKS = 1:param["N_SEGMENTS"]
    FUEL_PRICES = 1:param["N_FUELPRICES"]
    PROFILES = 1:param["N_PROFILES"]
    SHIFTERS = 1:param["N_SHIFTERS"]
    return sets(GENS, TIME_BLOCKS, FUEL_PRICES, PROFILES, SHIFTERS)
end

function init_parameters(param::Dict, set::sets)
    VOLL = param["VOLL"]
    DURATION = param["duration"]
    LOAD_FIX = Dict((t, r) => param["nominal_demand"][string(r)][t] for t in 1:param["N_SEGMENTS"], r in 1:param["N_PROFILES"])
    LOAD_FLEX = param["FLEX_DEMAND_MAX"]
    AVAILABILITY = Dict((g, r, t) => param["availability"][g][string(r)][t] for g in param["GENS"], t in 1:param["N_SEGMENTS"], r in 1:param["N_PROFILES"])
    MARG_COST = Dict((g, f) => param["fuel_price"][g][f] for g in param["GENS"], f in 1:param["N_FUELPRICES"])
    LOAD_SHIFT_NEG = param["fuel_response"]
    LOAD_SHIFT_POS = param["adder"]
    α = 0.7
    β = 0.6
    SCEN_PROB = Dict((f, r, s) => param["p_profile"][r] * param["p_adder"][s] * param["p_fuel_price"][f] for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    GEN_INV_COST = param["investment"]

    return parameters(VOLL, DURATION, LOAD_FIX, LOAD_FLEX,
    AVAILABILITY, MARG_COST, LOAD_SHIFT_NEG, LOAD_SHIFT_POS, α,
    β, SCEN_PROB, GEN_INV_COST)
end

INPUT_FPATH = "../pjm_wind.json"

input = JSON.parsefile(INPUT_FPATH)
set = init_sets(input)
param = init_parameters(input, set)

model = Model(CPLEX.Optimizer)

@variable(model, capacity[set.GENS] >= 0)
@variable(model, dem_served_fix[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] >=0)
@variable(model, dem_served_flex[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] >=0)
@variable(model, prod[set.GENS, set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] >=0)
@variable(model, surplus_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] >=0)
@variable(model, var)

@expression(model, surplus[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-sum(param.GEN_INV_COST[g] * capacity[g] for g in set.GENS)
    + sum(param.VOLL * param.DURATION[t] * (dem_served_fix[t,f,r,s] + dem_served_flex[t,f,r,s] - 0.5 * (dem_served_flex[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.MARG_COST[g,f] * param.DURATION[t] * prod[g,t,f,r,s] for g in set.GENS, t in set.TIME_BLOCKS))
    )

@objective(model, Max,
    ((1 - param.β) * (var - (1 / param.α) * (sum(param.SCEN_PROB[f,r,s] * surplus_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β * (sum(param.SCEN_PROB[f,r,s] * surplus[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    )

@constraint(model, risk_set[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var - surplus[f,r,s] <= surplus_aux[f,r,s])
    )

@constraint(model, energy_balance[t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (dem_served_fix[t,f,r,s] + dem_served_flex[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f] == sum(prod[g,t,f,r,s] for g in set.GENS))
    )

@constraint(model, max_prod[g in set.GENS, t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (prod[g,t,f,r,s] <= param.AVAILABILITY[g,r,t] * capacity[g]))

@constraint(model, max_dem_served_fix[t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (dem_served_fix[t,f,r,s] <= param.LOAD_FIX[t,r]))

@constraint(model, max_dem_served_flex[t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (dem_served_flex[t,f,r,s] <= param.LOAD_FLEX))

optimize!(model)


evalu = init_evaluation(input)
evalu.D_flex = param.LOAD_FLEX
evalu.α = param.α
evalu.β = param.β
evalu.CAPACITY = JuMP.value.(capacity)
evalu.SURPLUS = sum(param.SCEN_PROB[f,r,s] * JuMP.value.(surplus[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
evalu.Objective = objective_value(model)
evalu.VAR = JuMP.value.(var)
evalu.CVAR = JuMP.value.(var) - (1 / param.α) * (sum(param.SCEN_PROB[f,r,s] * JuMP.value.(surplus_aux[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
Total_prod = Dict((f,r,s) => sum(param.DURATION[t]* JuMP.value.(prod[g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
spot_price = Dict((f,r,s) => ((sum(shadow_price(model[:energy_balance][t,f,r,s])*param.DURATION[t]* sum(JuMP.value.(prod[g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
evalu.AVG_spot_price = mean(collect(values(spot_price)))
evalu.spot_price_std = std(collect(values(spot_price)))
evalu.Unserved_energy = sum(param.SCEN_PROB[f,r,s]* sum(param.DURATION[t]*(param.LOAD_FIX[t,r] - JuMP.value.(dem_served_fix[t,f,r,s])) for t in set.TIME_BLOCKS) for  f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

OUTPUT_FPATH = "../CPLEX_beta0.6.json"
open(OUTPUT_FPATH, "w") do f
    JSON.print(f, evalu,4)
end
