
def print_percentage_change(transf_results, orig_results):
    metrics_transf = transf_results['metrics']
    metrics_orig = orig_results['metrics']
    profit_transf = transf_results['profit']
    profit_orig = orig_results['profit']

    print("\nPercentage change for transformed data compared to original data:")

    for metric_name, metric_val in metrics_transf.items():
        orig_val = metrics_orig[metric_name]
        percent_change = ((metric_val - orig_val) / orig_val) * 100
        print(f"{metric_name}: {percent_change:.2f}%")

    # Profit percentage change
    orig_profit = profit_orig['profit']
    transf_profit = profit_transf['profit']
    profit_percent_change = ((transf_profit - orig_profit) / orig_profit) * 100
    print(f"\nProfit Percentage Change: {profit_percent_change:.2f}%")