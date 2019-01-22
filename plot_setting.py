import matplotlib.pyplot as plt

def plot(xl="x-axis", yl="y-axis"):
    def decorator(plot_func):
        def wrapper(*args, **kw):
            plt.figure(figsize=(10, 7.5))
            func = plot_func(*args, **kw)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.legend(fontsize=20)
            plt.xlabel(xl, fontsize=25)
            plt.ylabel(yl, fontsize=25)
            plt.show()
            return func
        return wrapper
    return decorator
