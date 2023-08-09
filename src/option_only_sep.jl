"""
option only (sell separately)

Module to replicate Table 1 and Table 2

Symbols in comments for set and parameter structs indicate the symbols used in the paper.

"""

using JSON, JuMP, Gurobi, DataStructures

struct sets
    GENS # g
    TIME_BLOCKS # t
    FUEL_PRICES # f
    PROFILES # r
    SHIFTERS # s
    CONTRACTS # c
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
    α_GEN # α_a
    α_CONS # α_c
    β_GEN # β_g
    β_CONS # β_c
    γ # γ
    SCEN_PROB # p_{frs}
    INV_COST # C_g^{INV}
end

mutable struct provisional_parameters
    ITERATION
    CAPACITY # x_q
    ENERGY_PRICE # λ_{frst}
    OPER_PROFIT # π_{fgrst}
    PAYOUT # η^k_{frs}
    CONTRACT_PRICE # ϕ^k
    DEM_SERVED_FIX # d_t^{fix}
    DEM_SERVED_FLEX # d_t^{flex}
    VOL_CONS # v_c^k
    VOL_GEN # v_g^k
    VOL_MIN_GEN # \underline{v}_g^k
    VOL_MIN_CONS # \underline{v}_c^k
    VOL_MAX_GEN # \overline{v}_g^k
    VOL_MAX_CONS # \overline{v}_c^k
    DUAL_RISK_SET_GEN
    DUAL_RISK_SET_CONS
    THETA
    CONS_SURPLUS
    CONS_objective
    GEN_objective
    AVG_Profit
    GEN_CVAR
    SCEN_PROFIT
end

mutable struct evaluation
    α_GEN # α_a
    α_CONS # α_c
    β_GEN # β_g
    β_CONS # β_c
    CAPACITY
    CONTRACT_PRICE
    VOL_GEN
    VOL_CON
    THETA_GEN
    THETA_CONS
    CONS_SURPLUS
    CONS_objective
    GEN_objective
    GEN_CVAR
    AVG_Profit
    PAYOUT
    DUAL_RISK_SET_GEN
    SCEN_PROFIT_NUC
end

function init_evaluation(param::Dict)
    α_GEN = Dict(g => 0.0 for g in param["GENS"])#α_a
    α_CONS = 0.0  # α_c
    β_GEN = Dict(g => 0.0 for g in param["GENS"])  # β_g
    β_CONS = 0.0  # β_c
    CAPACITY = Dict(g => 0.0 for g in param["GENS"])
    CONTRACT_PRICE = Dict()
    VOL_GEN = Dict()
    VOL_CON = Dict()
    THETA_GEN = Dict((g,c) => 0.0 for g in param["GENS"], c in param["CONTRACTS"])
    THETA_CONS = Dict(c => 0.0 for c in param["CONTRACTS"])
    CONS_SURPLUS = 0
    CONS_objective = 0
    GEN_objective = Dict(g => 0.0 for g in param["GENS"])
    GEN_CVAR = Dict(g => 0.0 for g in param["GENS"])
    AVG_Profit = Dict(g => 0.0 for g in param["GENS"])
    PAYOUT = Dict()
    DUAL_RISK_SET_GEN = Dict()
    SCEN_PROFIT_NUC = Dict((f,r,s) => 0.0 for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    return evaluation(α_GEN, α_CONS, β_GEN, β_CONS, CAPACITY, CONTRACT_PRICE, VOL_GEN, VOL_CON, THETA_GEN, THETA_CONS,
            CONS_SURPLUS, CONS_objective, GEN_objective, GEN_CVAR, AVG_Profit, PAYOUT, DUAL_RISK_SET_GEN, SCEN_PROFIT_NUC)
end

function init_sets(param::Dict)
    GENS = param["GENS"]
    TIME_BLOCKS = 1:param["N_SEGMENTS"]
    FUEL_PRICES = 1:param["N_FUELPRICES"]
    PROFILES = 1:param["N_PROFILES"]
    SHIFTERS = 1:param["N_SHIFTERS"]
    CONTRACTS = "OPTION_1000"  # Option_only
    return sets(GENS, TIME_BLOCKS, FUEL_PRICES, PROFILES, SHIFTERS, CONTRACTS)
end

function init_parameters(param::Dict)
    VOLL = param["VOLL"]
    DURATION = param["duration"]
    LOAD_FIX = Dict((t, r) => param["nominal_demand"][string(r)][t] for t in 1:param["N_SEGMENTS"], r in 1:param["N_PROFILES"])
    LOAD_FLEX = param["FLEX_DEMAND_MAX"]
    AVAILABILITY = Dict((g, r, t) => param["availability"][g][string(r)][t] for g in param["GENS"], t in 1:param["N_SEGMENTS"], r in 1:param["N_PROFILES"])
    MARG_COST = Dict((g, f) => param["fuel_price"][g][f] for g in param["GENS"], f in 1:param["N_FUELPRICES"])
    LOAD_SHIFT_NEG = param["fuel_response"]
    LOAD_SHIFT_POS = param["adder"]
    α_GEN = Dict(g => 0.7 for g in param["GENS"])
    α_CONS = 0.7
    β_GEN = Dict(g => 0.6 for g in param["GENS"])
    β_CONS = 0.7
    γ = 5.0
    SCEN_PROB = Dict((f, r, s) => param["p_profile"][r] * param["p_adder"][s] * param["p_fuel_price"][f] for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    INV_COST = param["investment"]

    return parameters(VOLL, DURATION, LOAD_FIX, LOAD_FLEX, AVAILABILITY, MARG_COST,
    LOAD_SHIFT_NEG, LOAD_SHIFT_POS, α_GEN, α_CONS, β_GEN, β_CONS,
    γ, SCEN_PROB, INV_COST)
end

function init_provisional_parameters(set::sets)
    ITERATION = 1
    CAPACITY = Dict("NUC" => 42.94, "CCGT" => 96.47, "WIND" => 129.0)

    ENERGY_PRICE = Dict()
    OPER_PROFIT = Dict()
    PAYOUT = Dict()
    CONTRACT_PRICE = 0.0
    DEM_SERVED_FIX = Dict()
    DEM_SERVED_FLEX = Dict()
    VOL_CONS = 0.0
    VOL_GEN = Dict(g => 0.0 for g in set.GENS)

    for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES
        PAYOUT[f,r,s] = 0.0
        for t in set.TIME_BLOCKS
            ENERGY_PRICE[t,f,r,s] = 0.0
            DEM_SERVED_FIX[t,f,r,s] = 0.0
            DEM_SERVED_FLEX[t,f,r,s] = 0.0

            for g in set.GENS
                OPER_PROFIT[g,t,f,r,s] = 0.0
            end
        end
    end

    VOL_MIN_GEN = Dict(g => 0.0 for g in set.GENS)
    VOL_MIN_CONS = 0.0
    VOL_MAX_GEN = Dict(g => 0.0 for g in set.GENS)
    VOL_MAX_CONS = 0.0
    DUAL_RISK_SET_GEN = Dict((g,f,r,s) =>  0.0 for g in set.GENS, r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    DUAL_RISK_SET_CONS = Dict((f,r,s) =>  0.0 for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    THETA = 0.0
    CONS_SURPLUS = 0.0
    CONS_objective = 0.0
    GEN_objective = Dict(g => 0.0 for g in set.GENS)
    AVG_Profit = Dict(g => 0.0 for g in set.GENS)
    GEN_CVAR = Dict(g => 0.0 for g in set.GENS)
    SCEN_PROFIT = Dict((g,f,r,s) =>  0.0 for g in set.GENS, r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)

    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT, CONTRACT_PRICE, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN, DUAL_RISK_SET_CONS,
    THETA, CONS_SURPLUS, CONS_objective, GEN_objective, AVG_Profit, GEN_CVAR, SCEN_PROFIT)
end

function make_dispatch_model(set::sets, param::parameters, prov_param::provisional_parameters)
    dispatch_model = Model(Gurobi.Optimizer)

    @variable(dispatch_model, 0 <= dem_served_fix[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(dispatch_model, 0 <= dem_served_flex[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] <= param.LOAD_FLEX)
    @variable(dispatch_model, 0 <= prod[set.GENS, set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])

    @objective(dispatch_model, Max,
    (sum(sum(param.VOLL * (dem_served_fix[t,f,r,s] + dem_served_flex[t,f,r,s] - 0.5 * (dem_served_flex[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.MARG_COST[g,f] * prod[g,t,f,r,s] for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    )

    @constraint(dispatch_model, energy_balance[t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (dem_served_fix[t,f,r,s] + dem_served_flex[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f] == sum(prod[g,t,f,r,s] for g in set.GENS))
    )

    @constraint(dispatch_model, max_prod[g in set.GENS, t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (prod[g,t,f,r,s] <= param.AVAILABILITY[g,r,t] * prov_param.CAPACITY[g])
    )

    @constraint(dispatch_model, max_dem_served_fix[t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (dem_served_fix[t,f,r,s] <= param.LOAD_FIX[t,r]))

    return dispatch_model
end

function make_cons_model(set::sets, param::parameters, prov_param::provisional_parameters)
    cons_model = Model(Gurobi.Optimizer)

    @variable(cons_model, prov_param.VOL_MIN_CONS <= vol_cons <= prov_param.VOL_MAX_CONS)
    @variable(cons_model, 0 <= scen_surplus_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(cons_model, var_cons)

    @expression(cons_model, scen_surplus[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (vol_cons* (prov_param.CONTRACT_PRICE - prov_param.PAYOUT[f,r,s])
    + sum(param.DURATION[t] * param.VOLL * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] - 0.5 * (prov_param.DEM_SERVED_FLEX[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.DURATION[t] * prov_param.ENERGY_PRICE[t,f,r,s] * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f]) for t in set.TIME_BLOCKS))
    )

    @objective(cons_model, Max,
    ((1 - param.β_CONS) * (var_cons - (1 / param.α_CONS) * (sum(param.SCEN_PROB[f,r,s] * scen_surplus_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_CONS * (sum(param.SCEN_PROB[f,r,s] * scen_surplus[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (vol_cons + sum(prov_param.VOL_GEN[g] for g in set.GENS))^2 )
    ) 

    @constraint(cons_model, risk_set_cons[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_cons - scen_surplus[f,r,s] <= scen_surplus_aux[f,r,s])
    )

    return cons_model
end

function make_gen_model(g, set::sets, param::parameters, prov_param::provisional_parameters)
    gen_model = Model(Gurobi.Optimizer)

    @variable(gen_model, prov_param.VOL_MIN_GEN[g] <= vol_gen <= prov_param.VOL_MAX_GEN[g])
    @variable(gen_model, 0 <= scen_profit_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(gen_model, var_gen)

    @expression(gen_model, scen_profit[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-param.INV_COST[g] * prov_param.CAPACITY[g]
    - vol_gen*(prov_param.CONTRACT_PRICE - prov_param.PAYOUT[f,r,s])
    + sum(prov_param.OPER_PROFIT[g,t,f,r,s] * prov_param.CAPACITY[g] for t in set.TIME_BLOCKS))
    ) 

    @objective(gen_model, Max,
    ((1 - param.β_GEN[g]) * (var_gen - (1 / param.α_GEN[g]) * (sum(param.SCEN_PROB[f,r,s] * scen_profit_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_GEN[g] * (sum(param.SCEN_PROB[f,r,s] * scen_profit[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (prov_param.VOL_CONS + vol_gen + sum(prov_param.VOL_GEN[q] for q in filter(x->x != g, set.GENS)))^2 )
    ) 

    @constraint(gen_model, risk_set_gen[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_gen - scen_profit[f,r,s] <= scen_profit_aux[f,r,s])
    )

    return gen_model
end


function init_input(input_path)
    param = JSON.parsefile(input_path)
    sets = init_sets(param)
    parameters = init_parameters(param)
    evaluation = init_evaluation(param)
    return sets, parameters, evaluation
end

function solve_dispatch(set, param, prov_param)
    dispatch_model = make_dispatch_model(set, param, prov_param)
    set_silent(dispatch_model)
    optimize!(dispatch_model)
    return dispatch_model
end

function solve_gen(set, param, prov_param)
    gen_model = Dict()
    for g in set.GENS
        gen_model[g] = make_gen_model(g, set, param, prov_param)
        set_silent(gen_model[g])
        optimize!(gen_model[g])
        prov_param.GEN_CVAR[g] = value(gen_model[g][:var_gen])
        for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
            prov_param.DUAL_RISK_SET_GEN[g,f,r,s] = shadow_price(gen_model[g][:risk_set_gen][f,r,s])
        end
    end
    return gen_model
end

function solve_cons(set, param, prov_param)
    cons_model = make_cons_model(set, param, prov_param)
    set_silent(cons_model)
    optimize!(cons_model)
    return cons_model
end

function calc_vol_limits(set, prov_param)
    vol_max_cons = prov_param.VOL_MAX_CONS
    vol_min_gen = prov_param.VOL_MIN_GEN

    vol_max_cons = 2.0 * sum(prov_param.CAPACITY[g] for g in set.GENS)
    for g in set.GENS
        vol_min_gen[g] = -2.0 * prov_param.CAPACITY[g]
    end
    return vol_max_cons, vol_min_gen
end

function calc_payout(set, param, prov_param, option_price = 1000.0)
    payout = prov_param.PAYOUT

    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        payout[f,r,s] = sum(param.DURATION[t] * max(0, prov_param.ENERGY_PRICE[t,f,r,s] - option_price) for t in set.TIME_BLOCKS)
    end

    return payout
end

function calc_gen_objective(set, param, gen_model)
    gen_objective = Dict()
    for g in set.GENS
        gen_objective[g] = ((1 - param.β_GEN[g]) * (value(gen_model[g][:var_gen]) - (1 / param.α_GEN[g]) * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit_aux][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
        + param.β_GEN[g] * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    end
    return gen_objective
end

function intermediate_prov_param(inte_param::Dict, set::sets)
    ITERATION = inte_param["ITERATION"]
    CAPACITY = Dict(g => inte_param["CAPACITY"][g] for g in set.GENS)
    ENERGY_PRICE = Dict((t,f,r,s) => inte_param["ENERGY_PRICE"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    OPER_PROFIT = Dict((g,t,f,r,s)=> inte_param["OPER_PROFIT"][string((g,t,f,r,s))] for g in set.GENS, t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    PAYOUT = Dict((f,r,s)=>inte_param["PAYOUT"][string((f,r,s))] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CONTRACT_PRICE = inte_param["CONTRACT_PRICE"]
    DEM_SERVED_FIX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FIX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    DEM_SERVED_FLEX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FLEX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    VOL_CONS = inte_param["VOL_CONS"]
    VOL_GEN = Dict(g => inte_param["VOL_GEN"][string(g)] for g in set.GENS)
    VOL_MIN_GEN = Dict(g => inte_param["VOL_MIN_GEN"][string(g)] for g in set.GENS)
    VOL_MIN_CONS = inte_param["VOL_MIN_CONS"]
    VOL_MAX_GEN = Dict(g => inte_param["VOL_MAX_GEN"][string(g)] for g in set.GENS)
    VOL_MAX_CONS = inte_param["VOL_MAX_CONS"]
    DUAL_RISK_SET_GEN =  Dict((g,f,r,s)=> inte_param["DUAL_RISK_SET_GEN"][string((g,f,r,s))] for g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    DUAL_RISK_SET_CONS = Dict((f,r,s)=> inte_param["DUAL_RISK_SET_CONS"][string((f,r,s))] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    THETA = inte_param["THETA"]
    CONS_SURPLUS = inte_param["CONS_SURPLUS"]
    CONS_objective = inte_param["CONS_objective"]
    GEN_objective = Dict(g => inte_param["GEN_objective"][g] for g in set.GENS)
    AVG_Profit = Dict(g => inte_param["AVG_Profit"][g] for g in set.GENS)
    GEN_CVAR = Dict(g => inte_param["GEN_CVAR"][g] for g in set.GENS)
    SCEN_PROFIT = Dict((g,f,r,s)=> inte_param["SCEN_PROFIT"][string((g,f,r,s))] for g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT, CONTRACT_PRICE, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN,
    DUAL_RISK_SET_CONS, THETA, CONS_SURPLUS, CONS_objective, GEN_objective, AVG_Profit, GEN_CVAR, SCEN_PROFIT)
end

function solve_equilibrium(set, param, initial)
    FPATH = "../option_only_sep/beta0.6/intermediate_equ.json"
    finish = false
    cons_surplus = nothing
    cons_obj = nothing
    step = 0.03
    ϵ = 1

    # iter = 1
    # prov_param = provisional_parameters
    # prov_param.GEN_objective = Dict(g => 0.0 for g in set.GENS)
    # GUROBI_ENV = Gurobi.Env()

        if initial ==1  #initialize parameters
                prov_param = init_provisional_parameters(set) # data passed to prov_param using function init_provisional_parameters
                open(FPATH, "w") do f
                    JSON.print(f, prov_param, 4)
                end
                open("../option_only_sep/beta0.6/max_risk_imbalance.json", "w") do f
                    println(f, "Running unrestricted trading ... ... ")
                end
                open("../option_only_sep/beta0.6/GEN_objective.json", "w") do f
                    println(f, "Running unrestricted trading ... ... ")
                end
            else
                inte_param = JSON.parsefile(FPATH) # read parameters from FPATH file
                prov_param = intermediate_prov_param(inte_param, set) # pass data to prov_param
            end

    while !finish && prov_param.ITERATION < 40000 
        println("Iteration: ", prov_param.ITERATION)

        prov_param.CAPACITY = Dict(g => max(0, prov_param.CAPACITY[g] + step * prov_param.GEN_objective[g] / param.INV_COST[g]) for g in set.GENS)

        prov_param.VOL_MAX_CONS, prov_param.VOL_MIN_GEN = calc_vol_limits(set, prov_param)

        dispatch_model = solve_dispatch(set, param, prov_param)

        for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
            prov_param.DEM_SERVED_FIX[t,f,r,s] = value(dispatch_model[:dem_served_fix][t,f,r,s])
            prov_param.DEM_SERVED_FLEX[t,f,r,s] = value(dispatch_model[:dem_served_flex][t,f,r,s])
            prov_param.ENERGY_PRICE[t,f,r,s] = shadow_price(dispatch_model[:energy_balance][t,f,r,s])
            for g in set.GENS
                prov_param.OPER_PROFIT[g,t,f,r,s] = shadow_price(dispatch_model[:max_prod][g,t,f,r,s]) * param.AVAILABILITY[g,r,t]* param.DURATION[t]
            end
        end

        prov_param.PAYOUT = calc_payout(set, param, prov_param)

        if prov_param.ITERATION == 1
            prov_param.CONTRACT_PRICE = sum(param.SCEN_PROB[f,r,s]*prov_param.PAYOUT[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
            prov_param.VOL_CONS = prov_param.VOL_MAX_CONS
            prov_param.VOL_GEN = Dict(g => -prov_param.VOL_CONS / length(set.GENS) for g in set.GENS)
        end

        cons_model = nothing
        gen_model = nothing

        imbalance = 1000
        while imbalance > ϵ
            println("Iteration: ", prov_param.ITERATION, " Imbalance: ", imbalance)
            println("Gen capacity: ", prov_param.CAPACITY)
            println("Gen objective: ", prov_param.GEN_objective)

            gen_model = solve_gen(set, param, prov_param)
            prov_param.VOL_GEN = Dict(g => value(gen_model[g][:vol_gen]) for g in set.GENS)

            cons_model = solve_cons(set, param, prov_param)
            prov_param.VOL_CONS = value(cons_model[:vol_cons])
            println("Consumer volume: ", prov_param.VOL_CONS)

            difference = prov_param.VOL_CONS + sum(prov_param.VOL_GEN[g] for g in set.GENS)

            prov_param.CONTRACT_PRICE = prov_param.CONTRACT_PRICE + param.γ * difference

            imbalance = maximum(map(x->abs(x), values(difference)))
            prov_param.GEN_objective = calc_gen_objective(set, param, gen_model)
        end

        prov_param.GEN_objective = calc_gen_objective(set, param, gen_model)
        for g in set.GENS
            for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
                prov_param.SCEN_PROFIT[g,f,r,s] = value(gen_model[g][:scen_profit][f,r,s])
            end
        end
        max_risk_imbalance = maximum(map(x->abs(x), values(prov_param.GEN_objective)))
        println("Max risk imbalance: ", max_risk_imbalance)
        open("../option_only_sep/beta0.6/max_risk_imbalance.json", "a") do f
            println(f, prov_param.ITERATION, " \t", max_risk_imbalance)
        end
        open("../option_only_sep/beta0.6/GEN_objective.json", "a") do f
            println(f, prov_param.ITERATION, " \t", prov_param.GEN_objective["NUC"],  " \t", prov_param.GEN_objective["CCGT"], " \t", prov_param.GEN_objective["WIND"])
        end


        if max_risk_imbalance < 40_000
            ϵ = 0.1
            step = 0.02
            if max_risk_imbalance < 5_000
                ϵ = 0.1
                step = 0.03
                if max_risk_imbalance < 1000
                    ϵ = 0.1
                    step = 0.03
                if max_risk_imbalance < 100
                    finish = true
                    prov_param.AVG_Profit = Dict(g => sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS) for g in set.GENS)
                    prov_param.CONS_SURPLUS = sum(param.SCEN_PROB[f,r,s] * value(cons_model[:scen_surplus][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    prov_param.CONS_objective = objective_value(cons_model)
                end
                end
            end
        end

        prov_param.ITERATION += 1
        open(FPATH, "w") do f
            JSON.print(f, prov_param,4)
        end
    end

    return prov_param
end

function solve_evaluation(set, param, prov_param, evalu)
    evalu.α_GEN = param.α_GEN# α_a
    evalu.α_CONS = param.α_CONS # α_c
    evalu.β_GEN = param.β_GEN # β_g
    evalu.β_CONS = param.β_CONS # β_c
    evalu.CAPACITY = prov_param.CAPACITY
    evalu.CONTRACT_PRICE = prov_param.CONTRACT_PRICE
    evalu.VOL_GEN = prov_param.VOL_GEN
    evalu.VOL_CON = prov_param.VOL_CONS
    evalu.CONS_SURPLUS = prov_param.CONS_SURPLUS
    evalu.CONS_objective = prov_param.CONS_objective
    evalu.GEN_objective = prov_param.GEN_objective
    evalu.GEN_CVAR = prov_param.GEN_CVAR
    evalu.AVG_Profit = prov_param.AVG_Profit
    evalu.PAYOUT = prov_param.PAYOUT
    evalu.DUAL_RISK_SET_GEN = prov_param.DUAL_RISK_SET_GEN

    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        evalu.SCEN_PROFIT_NUC[f,r,s] = prov_param.SCEN_PROFIT["NUC",f,r,s]
    end

    evalu.CAPACITY = SortedDict(evalu.CAPACITY )
    evalu.VOL_GEN = SortedDict(evalu.VOL_GEN)
    evalu.GEN_objective = SortedDict(evalu.GEN_objective)
    evalu.GEN_CVAR = SortedDict(evalu.GEN_CVAR)
    evalu.AVG_Profit = SortedDict(evalu.AVG_Profit)
    # evalu.PAYOUT = SortedDict(evalu.PAYOUT)
    evalu.DUAL_RISK_SET_GEN = SortedDict(evalu.DUAL_RISK_SET_GEN)
    evalu.SCEN_PROFIT_NUC = SortedDict(evalu.SCEN_PROFIT_NUC)
    return evalu
end

INPUT_FPATH = "../pjm_wind.json"

set, param, evalu = init_input(INPUT_FPATH)

prov_param = solve_equilibrium(set, param, 1)

evalu = solve_evaluation(set, param, prov_param, evalu)

OUTPUT_FPATH = "../option_only_sep/output_beta0.6.json"
open(OUTPUT_FPATH, "w") do f
    JSON.print(f, evalu,4)
end
