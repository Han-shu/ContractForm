"""
SFPFC from Wolak (2022) Long-term resource adequacy

Module to replicate Table 3 and Table 4

contract volume = vol_cons*8760
FORWARD PRICE = 50.0
payout[f,r,s] = sum(param.LOAD[f,s,t]*param.DURATION[t]*(prov_param.ENERGY_PRICE[t,f,r,s] - price) for t in set.TIME_BLOCKS)/param.YEAR_LOAD[f,s]

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
    AVG_Profit
    PAYOUT
end

function init_evaluation(param::Dict)
    D_flex = 0.0
    α_GEN = Dict(g => 0.0 for g in param["GENS"])#α_a
    α_CONS = 0.0  # α_c
    β_GEN = Dict(g => 0.0 for g in param["GENS"])  # β_g
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
    AVG_Profit = Dict(g => 0.0 for g in param["GENS"])
    PAYOUT = Dict()

    return evaluation(D_flex, α_GEN, α_CONS, β_GEN, β_CONS, CAPACITY, VOL_GEN, VOL_CON, CONS_SURPLUS, CONS_objective, GEN_objective,
    GEN_CVAR, CONTRACT_PRICE, AVG_PAYOUT, RISK_PREMIA, Unserved_energy, AVG_prod_price, AVG_risk_price, AVG_spot_price, AVG_cons_price, spot_price_std, CONS_price_std, AVG_Profit, PAYOUT)
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
    α_GEN = Dict(g => 0.7 for g in param["GENS"])
    α_CONS = 0.7
    β_GEN = Dict(g => 0.8 for g in param["GENS"])
    β_CONS = 0.7
    γ = 150.0
    SCEN_PROB = Dict((f, r, s) => param["p_profile"][r] * param["p_adder"][s] * param["p_fuel_price"][f] for r in 1:param["N_PROFILES"], s in 1:param["N_SHIFTERS"], f in 1:param["N_FUELPRICES"])
    INV_COST = param["investment"]
    LOAD = Dict((f,s,t) => LOAD_FIX[t,1] + LOAD_SHIFT_POS[s] - LOAD_SHIFT_NEG[f] for f in 1:param["N_FUELPRICES"], s in 1:param["N_SHIFTERS"], t in 1:param["N_SEGMENTS"])
    YEAR_LOAD = Dict((f,s) => sum(DURATION[t]*LOAD[f,s,t] for t in 1:param["N_SEGMENTS"]) for f in 1:param["N_FUELPRICES"], s in 1:param["N_SHIFTERS"])

    return parameters(VOLL, DURATION, LOAD_FIX, LOAD_FLEX, AVAILABILITY, MARG_COST,
    LOAD_SHIFT_NEG, LOAD_SHIFT_POS, α_GEN, α_CONS, β_GEN, β_CONS,
    γ, SCEN_PROB, INV_COST, LOAD, YEAR_LOAD)
end

function init_provisional_parameters(set::sets)
    ITERATION = 1
    CAPACITY =  Dict("NUC" => 88.9979128030505, "CCGT" => 72.55322849874335, "WIND" => 26.007879620171646)

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
    @variable(cons_model, scen_surplus_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS]>=0)
    @variable(cons_model, var_cons)

    @expression(cons_model, scen_surplus[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (- vol_cons*8760.0*(prov_param.CONTRACT_PRICE - prov_param.PAYOUT[f,r,s])
    + sum(param.DURATION[t] * param.VOLL * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] - 0.5 * (prov_param.DEM_SERVED_FLEX[t,f,r,s]^2) / param.LOAD_FLEX) for t in set.TIME_BLOCKS)
    - sum(param.DURATION[t] * prov_param.ENERGY_PRICE[t,f,r,s] * (prov_param.DEM_SERVED_FIX[t,f,r,s] + prov_param.DEM_SERVED_FLEX[t,f,r,s] + param.LOAD_SHIFT_POS[s] - param.LOAD_SHIFT_NEG[f]) for t in set.TIME_BLOCKS))
    )

    @objective(cons_model, Max,
    ((1 - param.β_CONS) * (var_cons - (1 / param.α_CONS) * (sum(param.SCEN_PROB[f,r,s] * scen_surplus_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_CONS * (sum(param.SCEN_PROB[f,r,s] * scen_surplus[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (vol_cons + sum(prov_param.VOL_GEN[g] for g in set.GENS))^2
    ))

    @constraint(cons_model, risk_set_cons[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (var_cons - scen_surplus[f,r,s] <= scen_surplus_aux[f,r,s])
    )

    return cons_model
end

function make_gen_model(g, set::sets, param::parameters, prov_param::provisional_parameters)
    gen_model = Model(Gurobi.Optimizer)

    @variable(gen_model, prov_param.VOL_MIN_GEN[g] <= vol_gen <= prov_param.VOL_MAX_GEN[g])
    @variable(gen_model, scen_profit_aux[set.FUEL_PRICES, set.PROFILES, set.SHIFTERS]>=0)
    @variable(gen_model, var_gen)

    @expression(gen_model, scen_profit[f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS],
    (-param.INV_COST[g] * prov_param.CAPACITY[g]
    - vol_gen*8760.0*(prov_param.CONTRACT_PRICE - prov_param.PAYOUT[f,r,s])
    + sum(prov_param.OPER_PROFIT[g,t,f,r,s] * prov_param.CAPACITY[g] for t in set.TIME_BLOCKS))
    ) # Mays AMPL model does not multiply by capacity and uses only marginal values. Multiplies OPER_PROFIT by duration here instead of in algorithm.
    # Q: Should OPER_PROFIT be multiplied with production instead of CAPACITY?

    @objective(gen_model, Max,
    ((1 - param.β_GEN[g]) * (var_gen - (1 / param.α_GEN[g]) * (sum(param.SCEN_PROB[f,r,s] * scen_profit_aux[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)))
    + param.β_GEN[g] * (sum(param.SCEN_PROB[f,r,s] * scen_profit[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS))
    - 0.5 * param.γ * (prov_param.VOL_CONS + vol_gen + sum(prov_param.VOL_GEN[q] for q in filter(x->x != g, set.GENS)))^2
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

function calc_payout(set, param, prov_param, price = 50.0)
    payout = prov_param.PAYOUT
    for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS
        payout[f,r,s] = sum(param.LOAD[f,s,t]*param.DURATION[t]*(prov_param.ENERGY_PRICE[t,f,r,s] - price) for t in set.TIME_BLOCKS)/param.YEAR_LOAD[f,s]
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
    VOL_MIN_CONS = 0.0
    VOL_MAX_GEN = Dict(g => inte_param["VOL_MAX_GEN"][string(g)] for g in set.GENS)
    VOL_MAX_CONS = 0.0
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
    FPATH = "../SFPFC_sep/beta0.8/intermediate_equ.json"
    finish = false
    cons_surplus = nothing
    cons_obj = nothing
    Total_operation_cost = nothing
    Total_prod = nothing
    cons_price = nothing
    spot_price = nothing
    step = 0.03
    temp_step = 0.0001
    ϵ = 5

        if initial ==1  #initialize parameters
                prov_param = init_provisional_parameters(set) # data passed to prov_param using function init_provisional_parameters
                open(FPATH, "w") do f
                    JSON.print(f, prov_param, 4)
                end
                open("../SFPFC_sep/beta0.8/max_risk_imbalance.json", "w") do f
                    println(f, "Running SFPFC Separate trading ... ... ")
                end
                open("../SFPFC_sep/beta0.8/GEN_objective.json", "w") do f
                    println(f, "Running SFPFC Separate trading ... ... ")
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
            prov_param.CONTRACT_PRICE = sum(param.SCEN_PROB[f,r,s]*(sum(param.DURATION[t]*prov_param.ENERGY_PRICE[t,f,r,s] for t in set.TIME_BLOCKS)/8760.0) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
            prov_param.VOL_CONS = prov_param.VOL_MAX_CONS
            prov_param.VOL_GEN = Dict(g => -prov_param.VOL_CONS / length(set.GENS) for g in set.GENS)
        end

        cons_model = nothing
        gen_model = nothing

        imbalance = 100
        difference = 100
        while imbalance > ϵ
            println("Iteration: ", prov_param.ITERATION, " Difference: ", difference)
            println("Gen capacity: ", prov_param.CAPACITY)
            println("Gen objective: ", prov_param.GEN_objective)
            gen_model = solve_gen(set, param, prov_param)
            prov_param.VOL_GEN = Dict(g => value(gen_model[g][:vol_gen]) for g in set.GENS)
            cons_model = solve_cons(set, param, prov_param)
            prov_param.VOL_CONS = value(cons_model[:vol_cons])
            println("Consumer volume: ", prov_param.VOL_CONS)
            println("Generator volume: ", prov_param.VOL_GEN)
            difference = prov_param.VOL_CONS + sum(prov_param.VOL_GEN[g] for g in set.GENS)
            imbalance = abs(difference)
            prov_param.CONTRACT_PRICE = prov_param.CONTRACT_PRICE + temp_step * difference
            println("The contract price is ", prov_param.CONTRACT_PRICE)

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
        open("../SFPFC_sep/beta0.8/max_risk_imbalance.json", "a") do f
            println(f, prov_param.ITERATION, " \t", max_risk_imbalance)
        end
        open("../SFPFC_sep/beta0.8/GEN_objective.json", "a") do f
            println(f, prov_param.ITERATION, " \t", prov_param.GEN_objective["NUC"],  " \t", prov_param.GEN_objective["CCGT"], " \t", prov_param.GEN_objective["WIND"])
        end

        # step = max_risk_imbalance * 0.5 * 10^-6

        # Mays AMPL implementation calculates max_risk_imbalance as gen_objective times capacity.
        if max_risk_imbalance < 10_000
            ϵ = 0.5
            step = 0.03
            if max_risk_imbalance < 5_000
                ϵ = 0.1
                step = 0.01
                if max_risk_imbalance < 10_00
                    ϵ = 0.2
                    step = 0.01
                if max_risk_imbalance < 100
                    finish = true
                    prov_param.AVG_Profit = Dict(g => sum(param.SCEN_PROB[f,r,s] * value(gen_model[g][:scen_profit][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS) for g in set.GENS)
                    prov_param.CONS_SURPLUS = sum(param.SCEN_PROB[f,r,s] * value(cons_model[:scen_surplus][f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    prov_param.CONS_objective = objective_value(cons_model)
                    Total_prod = Dict((f,r,s) => sum(param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    Total_operation_cost = Dict((f,r,s) => sum(param.MARG_COST[g,f]*param.DURATION[t]* value(dispatch_model[:prod][g,t,f,r,s]) for t in set.TIME_BLOCKS, g in set.GENS) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
                    cons_price = Dict((f,r,s) => ((sum(prov_param.ENERGY_PRICE[t,f,r,s]*param.DURATION[t]* sum(value(dispatch_model[:prod][g,t,f,r,s]) for g in set.GENS) for t in set.TIME_BLOCKS)+ prov_param.VOL_CONS*8760*(prov_param.CONTRACT_PRICE-prov_param.PAYOUT[f,r,s]))/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
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
    evalu.CONTRACT_PRICE = prov_param.CONTRACT_PRICE
    evalu.AVG_PAYOUT = sum(param.SCEN_PROB[f,r,s]*prov_param.PAYOUT[f,r,s] for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    evalu.RISK_PREMIA = evalu.CONTRACT_PRICE  - evalu.AVG_PAYOUT
    evalu.Unserved_energy = sum(param.SCEN_PROB[f,r,s]* sum(param.DURATION[t]*(param.LOAD_FIX[t,r] - prov_param.DEM_SERVED_FIX[t,f,r,s]) for t in set.TIME_BLOCKS) for  f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)

    prod_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS)+ Total_operation_cost[f,r,s]) /Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    risk_price = Dict((f,r,s) => ((sum(param.INV_COST[g]*prov_param.CAPACITY[g] for g in set.GENS) + Total_operation_cost[f,r,s] + prov_param.VOL_CONS*8760*(prov_param.CONTRACT_PRICE-prov_param.PAYOUT[f,r,s]) )/Total_prod[f,r,s]) for f in set.FUEL_PRICES, r in set.PROFILES, s in set.SHIFTERS)
    evalu.AVG_prod_price = mean(collect(values(prod_price)))
    evalu.AVG_risk_price  = mean(collect(values(risk_price)))
    evalu.AVG_spot_price = mean(collect(values(spot_price)))
    evalu.AVG_cons_price = mean(collect(values(cons_price)))
    evalu.spot_price_std = std(collect(values(spot_price)))
    evalu.CONS_price_std = std(collect(values(cons_price)))

    evalu.AVG_Profit = prov_param.AVG_Profit
    evalu.PAYOUT = prov_param.PAYOUT

    evalu.CAPACITY = SortedDict(evalu.CAPACITY)
    evalu.VOL_GEN = SortedDict(evalu.VOL_GEN)
    evalu.GEN_objective = SortedDict(evalu.GEN_objective)
    evalu.GEN_CVAR = SortedDict(evalu.GEN_CVAR)
    evalu.AVG_Profit = SortedDict(evalu.AVG_Profit)
    evalu.PAYOUT = SortedDict(evalu.PAYOUT)
    return evalu
end

INPUT_FPATH = "../pjm_wind.json"

set, param, evalu = init_input(INPUT_FPATH)

prov_param, Total_operation_cost, Total_prod, cons_price, spot_price = solve_equilibrium(set, param, 1)

evalu = solve_evaluation(set, param, prov_param, evalu)

OUTPUT_FPATH = "../SFPFC_sep/output_beta0.8.json"
open(OUTPUT_FPATH, "w") do f
    JSON.print(f, evalu,4)
end
