"""
SFPFC from Wolak (2022) Long-term resource adequacy

Module to replicate Table 3, Table 4


generators sell contract collectively
GEN and CONS
total volume: VOL_CONS*8760
GEN, CONS

"""


using JSON, JuMP, Gurobi, DataStructures, Statistics

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
    LOAD
    YEAR_LOAD
end

mutable struct provisional_parameters
    ITERATION
    CAPACITY # x_q
    ENERGY_PRICE # λ_{frst}
    OPER_PROFIT # π_{fgrst}
    PAYOUT_SF
    CONTRACT_PRICE_SF
    DEM_SERVED_FIX # d_t^{fix}
    DEM_SERVED_FLEX # d_t^{flex}
    VOL_CONS # v_c^k
    VOL_GEN # v_g^k
    VOL_MIN_GEN # \underline{v}_g^k
    VOL_MIN_CONS
    VOL_MAX_GEN # \overline{v}_g^k
    VOL_MAX_CONS
    DUAL_RISK_SET_GEN
    CONS_SURPLUS
    CONS_objective
    GEN_objective
    GEN_Profit
    GEN_CVAR
    SCEN_PROFIT
end

mutable struct evaluation
    D_flex
    α_GEN # α_a
    α_CONS # α_c
    β_GEN # β_g
    β_CONS # β_c
    CAPACITY
    VOL_GEN
    VOL_CON
    CONS_SURPLUS
    CONS_objective
    GEN_objective
    GEN_CVAR
    CONTRACT_PRICE_SF
    AVG_PAYOUT_SF
    RISK_PREMIA
    Unserved_energy
    AVG_prod_price
    AVG_risk_price
    AVG_spot_price
    AVG_cons_price
    spot_price_std
    CONS_price_std
    AVG_Profit
    SCEN_PROFIT
    PAYOUT_SF
end

function init_evaluation(param::Dict,set::sets)
    D_flex = 0.0
    α_GEN = 0.0 #α_a
    α_CONS = 0.0  # α_c
    β_GEN = 0.0   # β_g
    β_CONS = 0.0  # β_c
    CAPACITY = Dict(g => 0.0 for g in param["GENS"])
    VOL_GEN = 0.0
    VOL_CON = 0.0
    CONS_SURPLUS = 0
    CONS_objective = 0.0
    GEN_objective = 0.0
    GEN_CVAR = 0.0
    CONTRACT_PRICE_SF = 0.0
    AVG_PAYOUT_SF = 0.0
    RISK_PREMIA = 0.0
    Unserved_energy = 0.0
    AVG_prod_price = 0.0
    AVG_risk_price = 0.0
    AVG_spot_price = 0.0
    AVG_cons_price = 0.0
    spot_price_std = 0.0
    CONS_price_std = 0.0
    AVG_Profit = Dict(g => 0.0 for g in param["GENS"])
    SCEN_PROFIT = Dict((f,r,s) =>  0.0 for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    PAYOUT_SF = 0.0
    return evaluation(D_flex, α_GEN, α_CONS, β_GEN, β_CONS, CAPACITY, VOL_GEN, VOL_CON, CONS_SURPLUS, CONS_objective, GEN_objective, GEN_CVAR, CONTRACT_PRICE_SF,
    AVG_PAYOUT_SF, RISK_PREMIA, Unserved_energy, AVG_prod_price, AVG_risk_price, AVG_spot_price, AVG_cons_price, spot_price_std, CONS_price_std, AVG_Profit, SCEN_PROFIT, PAYOUT_SF)
end

function init_sets(param::Dict)
    GENS = param["GENS"]
    TIME_BLOCKS = 1:param["N_SEGMENTS"]
    FUEL_PRICES = 1:param["N_FUELPRICES"]
    PROFILES = 1:param["N_PROFILES"]
    SHIFTERS = 1:param["N_SHIFTERS"]
    CONTRACTS = param["CONTRACTS"]
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
    α_GEN = 0.7
    α_CONS = 0.7
    β_GEN = 0.8
    β_CONS = 0.7
    γ = 25.0
    SCEN_PROB = Dict((f, r, s) => param["p_profile"][r] * param["p_adder"][s] * param["p_fuel_price"][f] for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    INV_COST = param["investment"]
    LOAD = Dict((f,s,t) => LOAD_FIX[t,1] + LOAD_SHIFT_POS[s] - LOAD_SHIFT_NEG[f] for f in 1:param["N_FUELPRICES"], s in 1:param["N_SHIFTERS"], t in 1:param["N_SEGMENTS"])
    YEAR_LOAD = Dict((f,s) => sum(DURATION[t]*LOAD[f,s,t] for t in 1:param["N_SEGMENTS"]) for f in 1:param["N_FUELPRICES"], s in 1:param["N_SHIFTERS"])

    return parameters(VOLL, DURATION, LOAD_FIX, LOAD_FLEX, AVAILABILITY, MARG_COST,
    LOAD_SHIFT_NEG, LOAD_SHIFT_POS, α_GEN, α_CONS, β_GEN, β_CONS, γ, SCEN_PROB, INV_COST, LOAD, YEAR_LOAD)
end

function init_provisional_parameters(set::sets)
    ITERATION = 1
    # CAPACITY = Dict("NUC" => 75.6, "CCGT" => 79.6, "WIND" => 57.1) # β = 0.6 SFPFC sep
    # CAPACITY = Dict("NUC" => 84.1, "CCGT" => 75.0, "WIND" => 36.9) # β = 0.4 SFPFC sep
    # CAPACITY = Dict("NUC" => 89.0, "CCGT" => 72.5, "WIND" => 26.0) # β = 0.2 SFPFC sep
    CAPACITY =  Dict("NUC" => 52.9, "CCGT" => 92.8, "WIND" => 109.8)

    ENERGY_PRICE = Dict()
    OPER_PROFIT = Dict()
    PAYOUT_SF = Dict()
    CONTRACT_PRICE_SF = 0.0
    DEM_SERVED_FIX = Dict()
    DEM_SERVED_FLEX = Dict()
    VOL_CONS = 0.0
    VOL_GEN = Dict((g, c) => 0.0 for g in set.GENS, c in set.CONTRACTS)
    for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES
        PAYOUT_SF[f,r,s] = 0.0
        for t in set.TIME_BLOCKS
            ENERGY_PRICE[t,f,r,s] = 0.0
            DEM_SERVED_FIX[t,f,r,s] = 0.0
            DEM_SERVED_FLEX[t,f,r,s] = 0.0

            for g in set.GENS
                OPER_PROFIT[g,t,f,r,s] = 0.0
            end
        end
    end

    VOL_MIN_GEN = 0.0
    VOL_MIN_CONS = 0.0
    VOL_MAX_GEN = 0.0
    VOL_MAX_CONS = 0.0
    DUAL_RISK_SET_GEN = Dict((f,r,s) => 0.0 for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CONS_SURPLUS = 0.0
    CONS_objective = 0.0
    GEN_objective = 0.0
    GEN_Profit = Dict(g => 1.0 for g in set.GENS)
    GEN_CVAR = 0.0
    SCEN_PROFIT = Dict((f,r,s) =>  0.0 for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)

    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT_SF, CONTRACT_PRICE_SF, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN,
    CONS_SURPLUS, CONS_objective, GEN_objective, GEN_Profit, GEN_CVAR, SCEN_PROFIT)
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
    @variable(cons_model, scen_surplus_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS]>=0)
    @variable(cons_model, var_cons)

    @expression(cons_model, scen_surplus[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (- vol_cons*8760.0*(prov_param.CONTRACT_PRICE_SF - prov_param.PAYOUT_SF[f,r,s])
    + sum(param.DURATION[t] * param.VOLL * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] - 0.5 * (prov_param.DEM_SERVED_FLEX[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.DURATION[t] * prov_param.ENERGY_PRICE[t,f,r,s] * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f]) for t in set.TIME_BLOCKS))
    )

    @objective(cons_model, Max,
    ((1 - param.β_CONS) * (var_cons - (1 / param.α_CONS) * (sum(param.SCEN_PROB[f,r,s] * scen_surplus_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_CONS * (sum(param.SCEN_PROB[f,r,s] * scen_surplus[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    )) 

    @constraint(cons_model, risk_set_cons[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_cons - scen_surplus[f,r,s] <= scen_surplus_aux[f,r,s])
    )

    return cons_model
end

function make_gen_model(set::sets, param::parameters, prov_param::provisional_parameters)
    gen_model = Model(Gurobi.Optimizer)

    @variable(gen_model, prov_param.VOL_MIN_GEN <= vol_gen <= prov_param.VOL_MAX_GEN)
    @variable(gen_model, scen_profit_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS]>=0)
    @variable(gen_model, var_gen)

    @expression(gen_model, individual_profit[g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
        (-param.INV_COST[g] * prov_param.CAPACITY[g] + sum(prov_param.OPER_PROFIT[g,t,f,r,s] * prov_param.CAPACITY[g] for t in set.TIME_BLOCKS))
        )

    @expression(gen_model, scen_profit[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-sum(param.INV_COST[g] * prov_param.CAPACITY[g] for g in set.GENS)
    - vol_gen*8760*(prov_param.CONTRACT_PRICE_SF - prov_param.PAYOUT_SF[f,r,s])
    + sum(prov_param.OPER_PROFIT[g,t,f,r,s] * prov_param.CAPACITY[g] for t in set.TIME_BLOCKS, g in set.GENS))
    ) 

    @objective(gen_model, Max,
    ((1 - param.β_GEN) * (var_gen - (1 / param.α_GEN) * (sum(param.SCEN_PROB[f,r,s] * scen_profit_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_GEN * (sum(param.SCEN_PROB[f,r,s] * scen_profit[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (vol_gen + prov_param.VOL_CONS)^2
    ))

    @constraint(gen_model, risk_set_gen[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_gen - scen_profit[f,r,s] <= scen_profit_aux[f,r,s])
    )

    return gen_model
end

function init_input(input_path)
    param = JSON.parsefile(input_path)
    sets = init_sets(param)
    parameters = init_parameters(param)
    evaluation = init_evaluation(param,sets)
    return sets, parameters, evaluation
end

function solve_dispatch(set, param, prov_param)
    dispatch_model = make_dispatch_model(set, param, prov_param)
    set_silent(dispatch_model)
    optimize!(dispatch_model)
    return dispatch_model
end

function solve_gen(set, param, prov_param)
    gen_model = make_gen_model(set, param, prov_param)
    set_silent(gen_model)
    optimize!(gen_model)
    prov_param.GEN_CVAR = value(gen_model[:var_gen])
    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        prov_param.DUAL_RISK_SET_GEN[f,r,s] = shadow_price(gen_model[:risk_set_gen][f,r,s])
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
    vol_min_gen = -vol_max_cons

    return vol_max_cons, vol_min_gen
end


function calc_PAYOUT_SF(set, param, prov_param, price = 50.0)
    PAYOUT_SF = prov_param.PAYOUT_SF
    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        PAYOUT_SF[f,r,s] = sum(param.LOAD[f,s,t]*param.DURATION[t]*(prov_param.ENERGY_PRICE[t,f,r,s] - price) for t in set.TIME_BLOCKS)/param.YEAR_LOAD[f,s]
    end
    return PAYOUT_SF
end

function calc_gen_objective(set, param, gen_model)
    gen_objective = ((1 - param.β_GEN) * (value(gen_model[:var_gen]) - (1 / param.α_GEN) * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[:scen_profit_aux][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
        + param.β_GEN * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    return gen_objective
end

function calc_gen_profits(set, param, prov_param, gen_model)
    gen_profit = Dict()
    for g in set.GENS
        gen_profit[g] = sum((prov_param.DUAL_RISK_SET_GEN[f,r,s]+param.β_GEN*param.SCEN_PROB[f,r,s])*value(gen_model[:individual_profit][g,f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    end
    return gen_profit
end

function intermediate_prov_param(inte_param::Dict, set::sets)
    ITERATION = inte_param["ITERATION"]
    CAPACITY = Dict(g => inte_param["CAPACITY"][g] for g in set.GENS)
    ENERGY_PRICE = Dict((t,f,r,s) => inte_param["ENERGY_PRICE"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    OPER_PROFIT = Dict((g,t,f,r,s)=> inte_param["OPER_PROFIT"][string((g,t,f,r,s))] for g in set.GENS, t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    PAYOUT_SF = Dict((f,r,s)=>inte_param["PAYOUT_SF"][string((f,r,s))] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CONTRACT_PRICE_SF = inte_param["CONTRACT_PRICE_SF"]
    DEM_SERVED_FIX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FIX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    DEM_SERVED_FLEX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FLEX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    VOL_CONS = inte_param["VOL_CONS"]
    VOL_GEN = inte_param["VOL_GEN"]
    VOL_MIN_GEN = inte_param["VOL_MIN_GEN"]
    VOL_MIN_CONS = inte_param["VOL_MIN_CONS"]
    VOL_MAX_GEN = inte_param["VOL_MAX_GEN"]
    VOL_MAX_CONS = inte_param["VOL_MAX_CONS"]
    DUAL_RISK_SET_GEN = Dict((f,r,s)=> inte_param["DUAL_RISK_SET_GEN"][string((f,r,s))] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CONS_SURPLUS = inte_param["CONS_SURPLUS"]
    CONS_objective = inte_param["CONS_objective"]
    GEN_objective = inte_param["GEN_objective"]
    GEN_Profit = Dict(g => inte_param["GEN_Profit"][g] for g in set.GENS)
    GEN_CVAR = inte_param["GEN_CVAR"]
    SCEN_PROFIT = Dict((f,r,s)=> inte_param["SCEN_PROFIT"][string((f,r,s))] for g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT_SF, CONTRACT_PRICE_SF, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN,
    CONS_SURPLUS, CONS_objective, GEN_objective, GEN_Profit, GEN_CVAR, SCEN_PROFIT)
end


function solve_equilibrium(set, param, initial)
    FPATH = "../SFPFC_col/beta0.8/intermediate_equ.json"
    finish = false
    cons_surplus = nothing
    cons_obj = nothing
    Total_prod = nothing
    Total_operation_cost = nothing
    cons_price = nothing
    spot_price = nothing
    step = 0.02
    step_2 = 0.001
    ϵ = 5.0

        if initial ==1  #initialize parameters
                prov_param = init_provisional_parameters(set) # data passed to prov_param using function init_provisional_parameters
                open(FPATH, "w") do f
                    JSON.print(f, prov_param, 4)
                end
                open("../SFPFC_col/beta0.8/GEN_profit.json", "w") do f
                    println(f, "Running SFPFC collectively trading ... ... ")
                end
                open("../SFPFC_col/beta0.8/GEN_objective.json", "w") do f
                    println(f, "Running SFPFC collectively trading ... ... ")
                end
            else
                inte_param = JSON.parsefile(FPATH) # read parameters from FPATH file
                prov_param = intermediate_prov_param(inte_param, set) # pass data to prov_param
            end

    while !finish && prov_param.ITERATION < 40000 
        println("Iteration: ", prov_param.ITERATION)

        prov_param.CAPACITY = Dict(g => max(0, prov_param.CAPACITY[g] + step * prov_param.GEN_Profit[g] / param.INV_COST[g]) for g in set.GENS)

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

        # prov_param.PAYOUT = calc_payout(set, param, prov_param)

        if prov_param.ITERATION == 1
            prov_param.CONTRACT_PRICE_SF = sum(param.SCEN_PROB[f,r,s]*(sum(param.DURATION[t]*prov_param.ENERGY_PRICE[t,f,r,s] for t in set.TIME_BLOCKS)/8760.0) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
            prov_param.VOL_GEN = - 100.0
            prov_param.VOL_CONS = - prov_param.VOL_GEN
        end

        cons_model = nothing
        gen_model = nothing

        prov_param.PAYOUT_SF = calc_PAYOUT_SF(set, param, prov_param)
        imbalance = 1000
        flag = false
        while !flag
            println("Iteration: ", prov_param.ITERATION, " Imbalance: ", imbalance)
            println("Gen capacity: ", prov_param.CAPACITY)
            println("Gen objective: ", prov_param.GEN_objective)


            gen_model = solve_gen(set, param, prov_param)
            prov_param.VOL_GEN = value(gen_model[:vol_gen])

            cons_model = solve_cons(set, param, prov_param)
            prov_param.VOL_CONS = value(cons_model[:vol_cons])
            println("Consumer volume: ", prov_param.VOL_CONS)

            # theta_2 = sum((prov_param.DUAL_RISK_SET_GEN[f,r,s]+param.β_AGG*param.SCEN_PROB[f,r,s])*prov_param.PAYOUT_SF[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
            # difference_cons = prov_param.CONTRACT_PRICE_SF - theta_2

            difference_cons = prov_param.VOL_GEN + prov_param.VOL_CONS
            imbalance = abs(difference_cons)
            println("SFPFC CONTRACT PRICE is ", prov_param.CONTRACT_PRICE_SF)
            println("SFPFC imbalance: ", difference_cons)
            prov_param.GEN_Profit = calc_gen_profits(set,param, prov_param, gen_model)

            if imbalance < ϵ
                flag = true
                println("******** SFPFC imbalance is satisfied **********")
            else
                prov_param.CONTRACT_PRICE_SF = prov_param.CONTRACT_PRICE_SF + step_2*difference_cons
                println("Contract price: ", prov_param.CONTRACT_PRICE_SF)
                println("Cons imbalance: ", imbalance)
                println(prov_param.ITERATION, " \t", prov_param.GEN_Profit["NUC"],  " \t", prov_param.GEN_Profit["CCGT"], " \t", prov_param.GEN_Profit["WIND"])
            end

            prov_param.GEN_objective = calc_gen_objective(set, param, gen_model)

        end


        prov_param.GEN_objective = calc_gen_objective(set, param, gen_model)
        prov_param.GEN_Profit = calc_gen_profits(set,param, prov_param, gen_model)
        println(prov_param.ITERATION, " \t", prov_param.GEN_Profit["NUC"],  " \t", prov_param.GEN_Profit["CCGT"], " \t", prov_param.GEN_Profit["WIND"])
        for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
            prov_param.SCEN_PROFIT[f,r,s] = value(gen_model[:scen_profit][f,r,s])
        end
        # max_risk_imbalance = maximum(map(x->abs(x), values(prov_param.GEN_Profit)))
        max_risk_imbalance = abs(prov_param.GEN_objective)
        println("Gen objective: ", prov_param.GEN_objective)
        open("../SFPFC_col/beta0.8/GEN_profit.json", "a") do f
            println(f, prov_param.ITERATION, " \t", prov_param.GEN_Profit["NUC"],  " \t", prov_param.GEN_Profit["CCGT"], " \t", prov_param.GEN_Profit["WIND"])
        end
        open("../SFPFC_col/beta0.8/GEN_objective.json", "a") do f
            println(f, prov_param.ITERATION, " \t", prov_param.GEN_objective)
        end


        if max_risk_imbalance < 20_000
            ϵ = 0.5
            step = 0.02
            if max_risk_imbalance < 7_000
                ϵ = 0.3
                step = 0.01
                # if max_risk_imbalance < 5_00
                #     ϵ = 1.0
                #     step = 0.5
                if max_risk_imbalance < 100
                    finish = true
                    # prov_param.GEN_Profit = sum(param.SCEN_PROB[f,r,s] * value(gen_model[:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    prov_param.CONS_SURPLUS = sum(param.SCEN_PROB[f,r,s] * value(cons_model[:scen_surplus][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    prov_param.CONS_objective = objective_value(cons_model)
                    Total_prod = Dict((f,r,s) => sum(param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    Total_operation_cost = Dict((f,r,s) => sum(param.MARG_COST[g,f]*param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    cons_price = Dict((f,r,s) => ((sum(prov_param.ENERGY_PRICE[t,f,r,s]*param.DURATION[t]* sum(value(dispatch_model[:prod][g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS)+ prov_param.VOL_CONS*8760*(prov_param.CONTRACT_PRICE_SF-prov_param.PAYOUT_SF[f,r,s]))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    spot_price = Dict((f,r,s) => ((sum(prov_param.ENERGY_PRICE[t,f,r,s]*param.DURATION[t]* sum(value(dispatch_model[:prod][g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                end
                # end
            end
        end

        prov_param.ITERATION += 1
        open(FPATH, "w") do f
            JSON.print(f, prov_param,4)
        end
    end

    return prov_param, Total_prod, Total_operation_cost, cons_price, spot_price
end

function solve_evaluation(set, param, prov_param, evalu)
    evalu.D_flex = param.LOAD_FLEX
    evalu.α_GEN = param.α_GEN# α_a
    evalu.α_CONS = param.α_CONS # α_c
    evalu.β_GEN = param.β_GEN # β_g
    evalu.β_CONS = param.β_CONS # β_c
    evalu.CAPACITY = prov_param.CAPACITY
    evalu.VOL_GEN = prov_param.VOL_GEN
    evalu.VOL_CON = prov_param.VOL_CONS
    evalu.CONS_SURPLUS = prov_param.CONS_SURPLUS
    evalu.CONS_objective = prov_param.CONS_objective
    evalu.GEN_objective = prov_param.GEN_objective
    evalu.GEN_CVAR = prov_param.GEN_CVAR
    evalu.CONTRACT_PRICE_SF = prov_param.CONTRACT_PRICE_SF
    evalu.AVG_PAYOUT_SF = sum(param.SCEN_PROB[f,r,s]*prov_param.PAYOUT_SF[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    evalu.RISK_PREMIA = evalu.CONTRACT_PRICE_SF - evalu.AVG_PAYOUT_SF
    evalu.Unserved_energy = sum(param.SCEN_PROB[f,r,s]*sum(param.DURATION[t]*(param.LOAD_FIX[t,r]-prov_param.DEM_SERVED_FIX[t,f,r,s]) for t in set.TIME_BLOCKS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

    prod_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS)+ Total_operation_cost[f,r,s]) /Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    risk_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS) + Total_operation_cost[f,r,s] + prov_param.VOL_CONS*8760*(prov_param.CONTRACT_PRICE_SF-prov_param.PAYOUT_SF[f,r,s]) )/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

    evalu.AVG_prod_price = mean(collect(values(prod_price)))
    evalu.AVG_risk_price  = mean(collect(values(risk_price)))
    evalu.AVG_spot_price = mean(collect(values(spot_price)))
    evalu.AVG_cons_price = mean(collect(values(cons_price)))
    evalu.spot_price_std = std(collect(values(spot_price)))
    evalu.CONS_price_std = std(collect(values(cons_price)))

    evalu.SCEN_PROFIT = prov_param.SCEN_PROFIT
    evalu.PAYOUT_SF = prov_param.PAYOUT_SF
    evalu.CAPACITY = SortedDict(evalu.CAPACITY)
    evalu.SCEN_PROFIT = SortedDict(evalu.SCEN_PROFIT)
    evalu.PAYOUT_SF = SortedDict(evalu.PAYOUT_SF)
    return evalu
end

INPUT_FPATH = "../pjm_wind.json"

set, param, evalu = init_input(INPUT_FPATH)

prov_param, Total_prod, Total_operation_cost, cons_price, spot_price = solve_equilibrium(set, param, 1)

evalu = solve_evaluation(set, param, prov_param, evalu)

OUTPUT_FPATH = "../SFPFC_col/output_beta0.8.json"
open(OUTPUT_FPATH, "w") do f
    JSON.print(f, evalu,4)
end
