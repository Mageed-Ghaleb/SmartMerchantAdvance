
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def optimize_advance(volume_forecast):
    model = gp.Model("loan_optimization")
    amount = model.addVar(lb=1000, ub=50000, name="advance_amount")
    term = model.addVar(lb=3, ub=12, vtype=GRB.INTEGER, name="term_months")
    price = model.addVar(lb=0.05, ub=0.3, name="rate")

    expected_revenue = amount * price
    risk_penalty = (0.01 * (13 - term)) * amount
    model.setObjective(expected_revenue - risk_penalty, GRB.MAXIMIZE)

    model.addConstr(amount <= 0.5 * sum(volume_forecast), "volume_cap")

    model.optimize()
    return {
        'amount': amount.X,
        'term': term.X,
        'rate': price.X,
        'objective': model.ObjVal
    }

if __name__ == "__main__":
    forecast = [10000] * 12
    result = optimize_advance(forecast)
    print("Optimized Offer:", result)
