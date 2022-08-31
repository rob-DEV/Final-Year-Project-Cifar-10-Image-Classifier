def print_progress_bar(iteration, iteration_count, prefix="", length= 10):
    #rounding percentages 
    iteration += 1
    if len(prefix) > 0:
        prefix = "{0}:".format(prefix)

    percentage = (iteration / iteration_count)
    filled_bar_length = int(percentage * length)

    filled_progress = '█' * filled_bar_length
    unfilled_progress = '-' * (length - filled_bar_length - 1)
    line_end = '\r'

    if iteration == iteration_count:
        percentage = 1.0
        line_end = '\n'

    print("\r{0} {1}{2} {3:.1f}%".format(prefix, filled_progress, unfilled_progress, percentage * 100.0), end=line_end)
