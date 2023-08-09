"""
Mandatory trading with reduced mandated options 
mandated options = 148.5 - contribution of renewables

Module to replicate Table 1 and Table 2

Symbols in comments for set and parameter structs indicate the symbols used in the paper.

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
    CREDIT_WIND
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
    CONTRACT_PRICE
    AVG_PAYOUT
    RISK_PREMIA
    Unserved_energy
    AVG_prod_price
    AVG_risk_price
    AVG_spot_price
    AVG_cons_price
    spot_price_std
    CONS_price_std
    CREDIT_WIND
    AVG_Profit
    EXP_PROD
    UNIT_RISK_COST
    PAYOUT
end

function init_evaluation(param::Dict)
    D_flex = 0.0
    α_GEN = 0.0 #α_a
    α_CONS = 0.0  # α_c
    β_GEN = 0.0 # β_g
    β_CONS = 0.0  # β_c
    CAPACITY = Dict(g => 0.0 for g in param["GENS"])
    VOL_GEN = Dict()
    VOL_CON = Dict()
    CONS_SURPLUS = 0
    CONS_objective = 0
    GEN_objective = Dict(g => 0.0 for g in param["GENS"])
    GEN_CVAR = Dict(g => 0.0 for g in param["GENS"])
    CONTRACT_PRICE = Dict()
    AVG_PAYOUT = Dict(c => 0.0 for c in param["CONTRACTS"])
    RISK_PREMIA = Dict(c => 0.0 for c in param["CONTRACTS"])
    Unserved_energy = 0.0
    AVG_prod_price = 0.0
    AVG_risk_price = 0.0
    AVG_spot_price = 0.0
    AVG_cons_price = 0.0
    spot_price_std = 0.0
    CONS_price_std = 0.0
    CREDIT_WIND = 0.0
    AVG_Profit = Dict(g => 0.0 for g in param["GENS"])
    EXP_PROD = 0.0
    UNIT_RISK_COST = 0.0
    PAYOUT = Dict()

    return evaluation(D_flex, α_GEN, α_CONS, β_GEN, β_CONS, CAPACITY, VOL_GEN, VOL_CON, CONS_SURPLUS, CONS_objective, GEN_objective, GEN_CVAR,
    CONTRACT_PRICE, AVG_PAYOUT, RISK_PREMIA, Unserved_energy, AVG_prod_price, AVG_risk_price, AVG_spot_price, AVG_cons_price, spot_price_std, CONS_price_std, CREDIT_WIND, AVG_Profit, EXP_PROD, UNIT_RISK_COST, PAYOUT)
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

function init_parameters(param::Dict; β::Float64)
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
    β_GEN = β
    β_CONS = 0.7
    γ = 20.0
    SCEN_PROB = Dict((f, r, s) => param["p_profile"][r] * param["p_adder"][s] * param["p_fuel_price"][f] for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    INV_COST = param["investment"]

    return parameters(VOLL, DURATION, LOAD_FIX, LOAD_FLEX, AVAILABILITY, MARG_COST,
    LOAD_SHIFT_NEG, LOAD_SHIFT_POS, α_GEN, α_CONS, β_GEN, β_CONS, γ, SCEN_PROB, INV_COST)
end

function init_provisional_parameters(set::sets)
    ITERATION = 1

    # Initialize the capacity mix
    CAPACITY = Dict("NUC" => 58.3, "CCGT" => 88.9, "WIND" => 106.8)

    ENERGY_PRICE = Dict()
    OPER_PROFIT = Dict()
    PAYOUT = Dict()
    CONTRACT_PRICE = Dict(c => 0.0 for c in set.CONTRACTS)
    DEM_SERVED_FIX = Dict()
    DEM_SERVED_FLEX = Dict()
    VOL_CONS = Dict(c => 0.0 for c in set.CONTRACTS)
    VOL_GEN = Dict((g, c) => 0.0 for g in set.GENS, c in set.CONTRACTS)

    for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES
        for t in set.TIME_BLOCKS
            ENERGY_PRICE[t,f,r,s] = 0.0
            DEM_SERVED_FIX[t,f,r,s] = 0.0
            DEM_SERVED_FLEX[t,f,r,s] = 0.0

            for g in set.GENS
                OPER_PROFIT[g,t,f,r,s] = 0.0
            end
        end

        for c in set.CONTRACTS
            PAYOUT[c,f,r,s] = 0.0
        end
    end

    VOL_MIN_GEN = Dict((g, c) => 0.0 for g in set.GENS, c in set.CONTRACTS)
    VOL_MIN_CONS = Dict(c => 0.0 for c in set.CONTRACTS)
    VOL_MAX_GEN = Dict((g, c) => 0.0 for g in set.GENS, c in set.CONTRACTS)
    VOL_MAX_CONS = Dict(c => 0.0 for c in set.CONTRACTS)
    DUAL_RISK_SET_GEN = Dict((g,f,r,s) =>  0.0 for g in set.GENS, r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    DUAL_RISK_SET_CONS = Dict((f,r,s) =>  0.0 for r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    THETA = Dict(c => 0.0 for c in set.CONTRACTS)
    CONS_SURPLUS = 0.0
    CONS_objective = 0.0
    GEN_objective = Dict(g => 0.0 for g in set.GENS)
    AVG_Profit = Dict(g => 0.0 for g in set.GENS)
    GEN_CVAR = Dict(g => 0.0 for g in set.GENS)
    SCEN_PROFIT = Dict((g,f,r,s) =>  0.0 for g in set.GENS, r in set.PROFILES, s in set.SHIFTERS, f in set.FUEL_PRICES)
    CREDIT_WIND = 0.0
    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT, CONTRACT_PRICE, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN, DUAL_RISK_SET_CONS,
    THETA, CONS_SURPLUS, CONS_objective, GEN_objective, AVG_Profit, GEN_CVAR, SCEN_PROFIT, CREDIT_WIND)
end

function make_dispatch_model(set::sets, param::parameters, prov_param::provisional_parameters)
    dispatch_model = Model(Gurobi.Optimizer)

    @variable(dispatch_model, 0 <= dem_served_fix[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(dispatch_model, 0 <= dem_served_flex[set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS] <= param.LOAD_FLEX)
    @variable(dispatch_model, 0 <= prod[set.GENS, set.TIME_BLOCKS, set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])

    @objective(dispatch_model, Max,
    (sum(sum(param.DURATION[t]*param.VOLL * (dem_served_fix[t,f,r,s] + dem_served_flex[t,f,r,s] - 0.5 * (dem_served_flex[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.DURATION[t]*param.MARG_COST[g,f] * prod[g,t,f,r,s] for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
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

    @variable(cons_model, prov_param.VOL_MIN_CONS[c] <= vol_cons[c in set.CONTRACTS]  <= prov_param.VOL_MAX_CONS[c])
    @variable(cons_model, 0 <= scen_surplus_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(cons_model, var_cons)

    @constraint(cons_model, vol_cons["OPTION_1000"]== 148.5 + prov_param.CREDIT_WIND*prov_param.VOL_GEN["WIND","UNIT_CONT"])

    @expression(cons_model, scen_surplus[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-sum(vol_cons[c] * (prov_param.CONTRACT_PRICE[c] - prov_param.PAYOUT[c,f,r,s]) for c in set.CONTRACTS)
    + sum(param.DURATION[t] * param.VOLL * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] - 0.5 * (prov_param.DEM_SERVED_FLEX[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.DURATION[t] * prov_param.ENERGY_PRICE[t,f,r,s] * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f]) for t in set.TIME_BLOCKS))
    )

    @objective(cons_model, Max,
    ((1 - param.β_CONS) * (var_cons - (1 / param.α_CONS) * (sum(param.SCEN_PROB[f,r,s] * scen_surplus_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_CONS * (sum(param.SCEN_PROB[f,r,s] * scen_surplus[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    # - 0.5 * param.γ * (sum((vol_cons[c] + sum(prov_param.VOL_GEN[g,c] for g in set.GENS))^2 for c in set.CONTRACTS))
    ) # Mays AMPL model multiplies last term with sum of installed capacities.

    @constraint(cons_model, risk_set_cons[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_cons - scen_surplus[f,r,s] <= scen_surplus_aux[f,r,s])
    )

    return cons_model
end

function make_gen_model(g, set::sets, param::parameters, prov_param::provisional_parameters)
    gen_model = Model(Gurobi.Optimizer)

    @variable(gen_model, prov_param.VOL_MIN_GEN[g,c] <= vol_gen[c in set.CONTRACTS] <= prov_param.VOL_MAX_GEN[g,c])
    @variable(gen_model, 0 <= scen_profit_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS])
    @variable(gen_model, var_gen)

    @expression(gen_model, scen_profit[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-param.INV_COST[g] * prov_param.CAPACITY[g]
    - sum(vol_gen[c] * (prov_param.CONTRACT_PRICE[c] - prov_param.PAYOUT[c,f,r,s]) for c in set.CONTRACTS)
    + sum(prov_param.OPER_PROFIT[g,t,f,r,s] * prov_param.CAPACITY[g] for t in set.TIME_BLOCKS))
    ) 

    @objective(gen_model, Max,
    ((1 - param.β_GEN) * (var_gen - (1 / param.α_GEN) * (sum(param.SCEN_PROB[f,r,s] * scen_profit_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_GEN * (sum(param.SCEN_PROB[f,r,s] * scen_profit[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (sum((prov_param.VOL_CONS[c] + vol_gen[c] + sum(prov_param.VOL_GEN[q,c] for q in filter(x->x != g, set.GENS)))^2 for c in set.CONTRACTS)))
    ) 

    @constraint(gen_model, risk_set_gen[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_gen - scen_profit[f,r,s] <= scen_profit_aux[f,r,s])
    )

    return gen_model
end


function solve_credit_wind(set::sets, param::parameters, prov_param::provisional_parameters)
    credit_wind = 0.0
    index = 0
    for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        if prov_param.ENERGY_PRICE[t,f,r,s]>76
            index = index + 1
            credit_wind = credit_wind + param.AVAILABILITY["WIND",r,t]
        else
            continue
        end
    end
    if index == 0
        credit_wind = 1
    else
        credit_wind = credit_wind / index
    end
    println("The wind credit is: ", credit_wind)
    return credit_wind
end


function init_input(input_path; β::Float64)
    param = JSON.parsefile(input_path)
    sets = init_sets(param)
    parameters = init_parameters(param; β = β)
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

    for c in set.CONTRACTS
        if c == "UNIT_CONT"
            vol_max_cons[c] = 2.0 * prov_param.CAPACITY["WIND"]
        else
            vol_max_cons[c] = 2.0 * sum(prov_param.CAPACITY[g] for g in set.GENS)
        end
    end
    for g in set.GENS
        vol_min_gen[g,"FUTURE_50"] = -2.0 * prov_param.CAPACITY[g]
        vol_min_gen[g,"UNIT_CONT"] = -2.0 * prov_param.CAPACITY[g]
    end
    vol_min_gen["CCGT","OPTION_1000"] = - 0.9 * prov_param.CAPACITY["CCGT"]
    vol_min_gen["NUC","OPTION_1000"] = - 0.9 * prov_param.CAPACITY["NUC"]
    vol_min_gen["WIND","OPTION_1000"] = min(0, - prov_param.CREDIT_WIND * (prov_param.CAPACITY["WIND"] + prov_param.VOL_GEN["WIND","UNIT_CONT"]) -1)
    # vol_min_gen["WIND","OPTION_1000"] = - prov_param.CREDIT_WIND * prov_param.CAPACITY["WIND"]
    return vol_max_cons, vol_min_gen
end

function calc_payout(set, param, prov_param, option_price = 1000.0, future_price = 50.0, unit_cont_price = 50.0)
    payout = prov_param.PAYOUT

    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        payout["OPTION_1000",f,r,s] = sum(param.DURATION[t] * max(0, prov_param.ENERGY_PRICE[t,f,r,s] - option_price) for t in set.TIME_BLOCKS)
        payout["FUTURE_50",f,r,s] = sum(param.DURATION[t] * (min(option_price, prov_param.ENERGY_PRICE[t,f,r,s]) - future_price) for t in set.TIME_BLOCKS)
        payout["UNIT_CONT",f,r,s] = sum(param.DURATION[t] * param.AVAILABILITY["WIND",r,t] * (min(option_price, prov_param.ENERGY_PRICE[t,f,r,s]) - unit_cont_price) for t in set.TIME_BLOCKS)
    end

    return payout
end

function calc_gen_objective(set, param, gen_model)
    gen_objective = Dict()
    for g in set.GENS
        gen_objective[g] = ((1 - param.β_GEN) * (value(gen_model[g][:var_gen]) - (1 / param.α_GEN) * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit_aux][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
        + param.β_GEN * (sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    end
    return gen_objective
end

function intermediate_prov_param(inte_param::Dict, set::sets)
    ITERATION = inte_param["ITERATION"]
    CAPACITY = Dict(g => inte_param["CAPACITY"][g] for g in set.GENS)
    ENERGY_PRICE = Dict((t,f,r,s) => inte_param["ENERGY_PRICE"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    OPER_PROFIT = Dict((g,t,f,r,s)=> inte_param["OPER_PROFIT"][string((g,t,f,r,s))] for g in set.GENS, t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    PAYOUT = Dict((c,f,r,s)=>inte_param["PAYOUT"][string((c,f,r,s))] for c in set.CONTRACTS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CONTRACT_PRICE = Dict(c => inte_param["CONTRACT_PRICE"][c] for c in set.CONTRACTS)
    DEM_SERVED_FIX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FIX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    DEM_SERVED_FLEX = Dict((t,f,r,s)=> inte_param["DEM_SERVED_FLEX"][string((t,f,r,s))] for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    VOL_CONS = Dict(c => inte_param["VOL_CONS"][c] for c in set.CONTRACTS)
    VOL_GEN = Dict((g, c) => inte_param["VOL_GEN"][string((g,c))] for g in set.GENS, c in set.CONTRACTS)
    VOL_MIN_GEN = Dict((g, c) => inte_param["VOL_MIN_GEN"][string((g,c))] for g in set.GENS, c in set.CONTRACTS)
    VOL_MIN_CONS = Dict(c => inte_param["VOL_MIN_CONS"][c] for c in set.CONTRACTS)
    VOL_MAX_GEN = Dict((g, c) => inte_param["VOL_MAX_GEN"][string((g,c))] for g in set.GENS, c in set.CONTRACTS)
    VOL_MAX_CONS = Dict(c => inte_param["VOL_MAX_CONS"][c] for c in set.CONTRACTS)
    DUAL_RISK_SET_GEN =  Dict((g,f,r,s)=> inte_param["DUAL_RISK_SET_GEN"][string((g,f,r,s))] for g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    DUAL_RISK_SET_CONS = Dict((f,r,s)=> inte_param["DUAL_RISK_SET_CONS"][string((f,r,s))] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    THETA = Dict(c => inte_param["THETA"][c] for c in set.CONTRACTS)
    CONS_SURPLUS = inte_param["CONS_SURPLUS"]
    CONS_objective = inte_param["CONS_objective"]
    GEN_objective = Dict(g => inte_param["GEN_objective"][g] for g in set.GENS)
    AVG_Profit = Dict(g => inte_param["AVG_Profit"][g] for g in set.GENS)
    GEN_CVAR = Dict(g => inte_param["GEN_CVAR"][g] for g in set.GENS)
    SCEN_PROFIT = Dict((g,f,r,s)=> inte_param["SCEN_PROFIT"][string((g,f,r,s))] for g in set.GENS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    CREDIT_WIND = inte_param["CREDIT_WIND"]
    return provisional_parameters(ITERATION, CAPACITY, ENERGY_PRICE, OPER_PROFIT, PAYOUT, CONTRACT_PRICE, DEM_SERVED_FIX,
    DEM_SERVED_FLEX, VOL_CONS, VOL_GEN, VOL_MIN_GEN, VOL_MIN_CONS, VOL_MAX_GEN, VOL_MAX_CONS, DUAL_RISK_SET_GEN,
    DUAL_RISK_SET_CONS, THETA, CONS_SURPLUS, CONS_objective, GEN_objective, AVG_Profit, GEN_CVAR, SCEN_PROFIT, CREDIT_WIND)
end


function solve_equilibrium(set, param, initial; β::Float64)

    FPATH = "reduced_mandatory/beta$β/intermediate_equ.json"
    finish = false
    cons_surplus = nothing
    cons_obj = nothing
    Total_prod = nothing
    Total_operation_cost = nothing
    cons_price = nothing
    spot_price = nothing
    step = 0.02
    temp_step = 1e-3
    ϵ = 0.2

    if initial == 1  #initialize parameters
            prov_param = init_provisional_parameters(set) # data passed to prov_param using function init_provisional_parameters
            open(FPATH, "w") do f
                JSON.print(f, prov_param, 4)
            end
            open("reduced_mandatory/beta$β/max_risk_imbalance.json", "w") do f
                println(f, "Running reduced mandatory trading... ... ")
            end
            open("reduced_mandatory/beta$β/GEN_objective.json", "w") do f
                println(f, "Running reduced mandatory trading... ... ")
            end
    else
        inte_param = JSON.parsefile(FPATH) # read parameters from FPATH file
        prov_param = intermediate_prov_param(inte_param, set) # pass data to prov_param
    end

    while !finish && prov_param.ITERATION < 20000 
        println("Iteration: ", prov_param.ITERATION)

        prov_param.CAPACITY = Dict(g => max(0, prov_param.CAPACITY[g] + step * prov_param.GEN_objective[g] / param.INV_COST[g]) for g in set.GENS)
        dispatch_model = solve_dispatch(set, param, prov_param)

        for t in set.TIME_BLOCKS, f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
            prov_param.DEM_SERVED_FIX[t,f,r,s] = value(dispatch_model[:dem_served_fix][t,f,r,s])
            prov_param.DEM_SERVED_FLEX[t,f,r,s] = value(dispatch_model[:dem_served_flex][t,f,r,s])
            prov_param.ENERGY_PRICE[t,f,r,s] = shadow_price(dispatch_model[:energy_balance][t,f,r,s])/param.DURATION[t]
            for g in set.GENS
                prov_param.OPER_PROFIT[g,t,f,r,s] = shadow_price(dispatch_model[:max_prod][g,t,f,r,s]) * param.AVAILABILITY[g,r,t]
            end
        end

        # Calaculate the marginal contribution of wind
        prov_param.CREDIT_WIND = solve_credit_wind(set, param, prov_param)
        prov_param.VOL_MAX_CONS, prov_param.VOL_MIN_GEN = calc_vol_limits(set, prov_param)

        prov_param.PAYOUT = calc_payout(set, param, prov_param)

        if prov_param.ITERATION == 1
            prov_param.CONTRACT_PRICE = Dict(c => sum(param.SCEN_PROB[f,r,s] * prov_param.PAYOUT[c,f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS) for c in set.CONTRACTS)
            prov_param.VOL_CONS = Dict(c => prov_param.VOL_MAX_CONS[c] / max(length(set.CONTRACTS)) for c in set.CONTRACTS)
            prov_param.VOL_GEN = Dict((g, c) => -prov_param.VOL_CONS[c] / length(set.GENS) for g in set.GENS, c in set.CONTRACTS)
        end

        cons_model = nothing
        gen_model = nothing

        imbalance = 1000
        while imbalance > ϵ
            println("Iteration: ", prov_param.ITERATION, " Imbalance: ", imbalance)
            println("Gen capacity: ", prov_param.CAPACITY)
            println("Gen objective: ", prov_param.GEN_objective)

            gen_model = solve_gen(set, param, prov_param)
            prov_param.VOL_GEN = Dict((g, c) => value(gen_model[g][:vol_gen][c]) for g in set.GENS, c in set.CONTRACTS)
            println("Generator volume: ", prov_param.VOL_GEN)

            cons_model = solve_cons(set, param, prov_param)
            prov_param.VOL_CONS = Dict(c => value(cons_model[:vol_cons][c]) for c in set.CONTRACTS)
            println("Consumer volume: ", prov_param.VOL_CONS)

            difference = Dict(c => (prov_param.VOL_CONS[c] + sum(prov_param.VOL_GEN[g,c] for g in set.GENS)) for c in set.CONTRACTS)
            println("Difference is: ", difference)
            imbalance = maximum(map(x->abs(x), values(difference)))
            if imbalance < 100
                temp_step = 0.5
            if imbalance < 30
                temp_step = 5.0
                if imbalance < 20
                    temp_step = 10.0
                    if imbalance < 10
                        temp_step = 10.0
                    end
                end
            end
            end
            prov_param.CONTRACT_PRICE = Dict(c => prov_param.CONTRACT_PRICE[c] + temp_step * difference[c] for c in set.CONTRACTS)
            println("CONTRACT PRICE: ", prov_param.CONTRACT_PRICE)

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
        open("reduced_mandatory/beta$β/max_risk_imbalance.json", "a") do f
            println(f, prov_param.ITERATION, " \t", max_risk_imbalance)
        end
        open("reduced_mandatory/beta$β/GEN_objective.json", "a") do f
            println(f, prov_param.ITERATION, " \t", prov_param.GEN_objective["NUC"],  " \t", prov_param.GEN_objective["CCGT"], " \t", prov_param.GEN_objective["WIND"])
        end


        if max_risk_imbalance < 5_0000
            ϵ = 0.05
            step = 0.01
            if max_risk_imbalance < 5_000
                ϵ = 0.01
                step = 0.05
                if max_risk_imbalance < 10_00
                    ϵ = 0.01
                    step = 0.01
                if max_risk_imbalance < 120
                    finish = true
                    prov_param.AVG_Profit = Dict(g => sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS) for g in set.GENS)
                    prov_param.CONS_SURPLUS = sum(param.SCEN_PROB[f,r,s] * value(cons_model[:scen_surplus][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    prov_param.CONS_objective = objective_value(cons_model)
                    Total_prod = Dict((f,r,s) => sum(param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    Total_operation_cost = Dict((f,r,s) => sum(param.MARG_COST[g,f]*param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    cons_price = Dict((f,r,s) => ((sum(prov_param.ENERGY_PRICE[t,f,r,s]*param.DURATION[t]* sum(value(dispatch_model[:prod][g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS) + sum(prov_param.VOL_CONS[c]*(prov_param.CONTRACT_PRICE[c]-prov_param.PAYOUT[c,f,r,s]) for c in set.CONTRACTS))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    spot_price = Dict((f,r,s) => ((sum(prov_param.ENERGY_PRICE[t,f,r,s]*param.DURATION[t]* sum(value(dispatch_model[:prod][g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                end
                end
            end
        end

        prov_param.ITERATION += 1
        open(FPATH, "w") do f
            JSON.print(f, prov_param,4)
        end
    end

    return prov_param, Total_operation_cost, Total_prod, cons_price, spot_price
end

function solve_evaluation(set, param, prov_param, evalu, Total_operation_cost, Total_prod, cons_price, spot_price)
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
    evalu.CONTRACT_PRICE = prov_param.CONTRACT_PRICE
    evalu.AVG_PAYOUT = Dict(c=> sum(param.SCEN_PROB[f,r,s]*prov_param.PAYOUT[c,f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS) for c in set.CONTRACTS)
    evalu.RISK_PREMIA = Dict(c => (evalu.CONTRACT_PRICE[c] - evalu.AVG_PAYOUT[c]) for c in set.CONTRACTS)
    evalu.Unserved_energy = sum(param.SCEN_PROB[f,r,s]*sum(param.DURATION[t]*(param.LOAD_FIX[t,r]-prov_param.DEM_SERVED_FIX[t,f,r,s]) for t in set.TIME_BLOCKS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    prod_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS)+ Total_operation_cost[f,r,s]) /Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    risk_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS) + Total_operation_cost[f,r,s] + sum(prov_param.VOL_CONS[c]*(prov_param.CONTRACT_PRICE[c]-prov_param.PAYOUT[c,f,r,s]) for c in set.CONTRACTS))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    evalu.AVG_prod_price = mean(collect(values(prod_price)))
    evalu.AVG_risk_price  = mean(collect(values(risk_price)))
    evalu.AVG_spot_price = mean(collect(values(spot_price)))
    evalu.AVG_cons_price = mean(collect(values(cons_price)))
    evalu.spot_price_std = std(collect(values(spot_price)))
    evalu.CONS_price_std = std(collect(values(cons_price)))

    evalu.CREDIT_WIND = prov_param.CREDIT_WIND
    evalu.AVG_Profit = prov_param.AVG_Profit
    evalu.EXP_PROD = mean(collect(values(Total_prod)))
    evalu.UNIT_RISK_COST = sum(param.SCEN_PROB[f,r,s]* (sum(prov_param.VOL_CONS[c]*(prov_param.CONTRACT_PRICE[c]-prov_param.PAYOUT[c,f,r,s]) for c in set.CONTRACTS)/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    evalu.PAYOUT = prov_param.PAYOUT

    evalu.CAPACITY = SortedDict(evalu.CAPACITY)
    evalu.VOL_GEN = SortedDict(evalu.VOL_GEN)
    evalu.VOL_CON = SortedDict(evalu.VOL_CON)
    evalu.GEN_objective = SortedDict(evalu.GEN_objective)
    evalu.GEN_CVAR = SortedDict(evalu.GEN_CVAR)
    evalu.CONTRACT_PRICE = SortedDict(evalu.CONTRACT_PRICE)
    evalu.AVG_PAYOUT = SortedDict(evalu.AVG_PAYOUT)
    evalu.RISK_PREMIA = SortedDict(evalu.RISK_PREMIA)
    evalu.AVG_Profit = SortedDict(evalu.AVG_Profit)
    evalu.PAYOUT = SortedDict(evalu.PAYOUT)
    return evalu
end


β = 0.8

set, param, evalu = init_input("pjm_wind.json"; β = β)

prov_param, Total_operation_cost, Total_prod, cons_price, spot_price = solve_equilibrium(set, param, 0; β = β)

evalu = solve_evaluation(set, param, prov_param, evalu, Total_operation_cost, Total_prod, cons_price, spot_price)

OUTPUT_FPATH ="reduced_mandatory/output_beta$β.json"
open(OUTPUT_FPATH, "w") do f
    JSON.print(f, evalu,4)
end
