
def euclidian_attractor_function_generator(n, max_positive=1.0, max_negative=1.0):
    def n_dimensional_euclid_function(x):
        ((max_positive+max_negative)/((x+1)**2))-max_negative
