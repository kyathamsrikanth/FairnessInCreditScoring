

import numpy as np
def compute_profit(class_preds, targets, amounts, r=0.2644):
    # Placeholder
    loan_profit = []

    # Go through loan applications
    for i in range(len(targets)):
        # Label and target
        pred_label = class_preds[i]
        true_label = targets[i]
        amount = amounts[i] * 1000
        p=0
        # Compute profit
        # bad & bad
        if pred_label == 2.0 and true_label == 2.0:
            p = 0
        # good & bad
        elif pred_label == 1.0 and true_label == 2.0:
            p = -0.25 * amount
        # good & good
        elif pred_label == 1.0 and true_label == 1.0:
            p = amount * r
        # bad & good
        elif pred_label == 2.0 and true_label == 1.0:
            p = -amount * r
        loan_profit.append(p)

    # Total profit
    profit = sum(loan_profit)
    profit_per_loan = profit / len(targets)
    profit_per_eur = profit_per_loan / np.mean(amounts)

    # Output
    return {"profit": profit, "profitPerLoan": profit_per_loan, "profitPerEUR": profit_per_eur}