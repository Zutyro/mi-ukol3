
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.ticker import FuncFormatter
import multiprocessing


def test_domain(point_values, domain):
    for x in point_values:
        if x < domain[0] or x > domain[1]:
            return False
    return True

def sphere(point_values):#Funkce cislo 0
    sum1 = 0
    for x in point_values:
        sum1 = sum1 + (x ** 2)
    y = sum1
    return y

def schwefel(point_values):#Funkce cislo 1
    sum1 = 0
    for x in point_values:
        sum1 = sum1 + x * np.sin(np.sqrt(np.abs(x)))
    y = 418.9829 * len(point_values) - sum1
    return y

def dixonprice(point_values):#Funkce cislo 2
    sum1 = 0
    for j, x in enumerate(point_values):
        if j == 0:
            continue
        sum1 = sum1 + (j + 1) * (2 * (x ** 2) - point_values[j - 1]) ** 2
    y = (point_values[0] - 1) ** 2 + sum1
    return y





def generate_coords(domain, center_point, standard_deviation, dimensions):
    coords = []
    for j in range(dimensions):
        while True:
            generated = random.gauss(center_point[j], standard_deviation)
            if domain[0] <= generated <= domain[1]:
                coords.append(generated)
                break
    return coords



def search(max_iterations: int, dimensions: int, population: int, standard_deviation: float, function_type: int, search_alg='hc'):
    if search_alg != 'hc' and search_alg != 'ls':
        raise ValueError('search_alg can only be hc or ls')
    domain = []
    match function_type:
        case 0:
            function_name = 'sphere'
            test_function = sphere
            domain = [-5.12,5.12]
        case 1:
            function_name ='schwefel'
            test_function = schwefel
            domain = [-500,500]
        case 2:
            function_name ='dixonprice'
            test_function = dixonprice
            domain = [-10,10]
        case _:
            raise ValueError('Function_type outside of range. Accepts values between 0 and 4')
    center_point = [random.uniform(domain[0], domain[1]) for i in range(dimensions)]
    center_result = test_function(center_point)
    best_result = 0
    best_point = 0
    convergence_points = []
    convergence_results = []
    convergence_iteration = []
    climb_results = []
    climb_points = []
    for i in range(max_iterations):
        best_result = 0
        best_point = 0
        for x in range(population):
            coords = generate_coords(domain, center_point, standard_deviation, dimensions)
            local_test = test_function(coords)
            if search_alg == 'hc' and best_result == 0:
                best_result = local_test
                best_point = coords
            if search_alg == 'ls' and best_result == 0:
                best_result = center_result
                best_point = center_point
            if local_test < best_result:
                best_result = local_test
                best_point = coords
        center_point = best_point
        climb_points.append(best_point)
        climb_results.append(best_result)
        if len(convergence_points) == 0:
            convergence_points.append(best_point)
            convergence_results.append(best_result)
        elif convergence_results[len(convergence_points) - 1] > best_result:
            convergence_points.append(best_point)
            convergence_results.append(best_result)
        else:
            convergence_results.append(convergence_results[len(convergence_results) - 1])
            convergence_points.append(convergence_points[len(convergence_points) - 1])
    return convergence_results


def simulated_annealing(temp0: int,tempF: int, dimensions: int, population: int, standard_deviation: float, function_type: int):
    domain = []
    match function_type:
        case 0:
            function_name = 'sphere'
            test_function = sphere
            domain = [-5.12, 5.12]
        case 1:
            function_name = 'schwefel'
            test_function = schwefel
            domain = [-500, 500]
        case 2:
            function_name = 'dixonprice'
            test_function = dixonprice
            domain = [-10, 10]
        case _:
            raise ValueError('Function_type outside of range. Accepts values between 0 and 2')



    center_point = [random.uniform(domain[0], domain[1]) for i in range(dimensions)]
    center_result = test_function(center_point)
    best_result = 0
    best_point = 0
    convergence_points = []
    convergence_results = []
    convergence_iteration = []
    climb_results = []
    climb_points = []

    temp = temp0
    cooling_rate = (temp0 - tempF)/(10000*dimensions/population)
    while temp > tempF:
        best_result = center_result
        best_point = center_point
        for x in range(population):
            coords = generate_coords(domain, center_point, standard_deviation, dimensions)
            local_test = test_function(coords)
            if local_test < best_result:
                best_result = local_test
                best_point = coords
            acceptance_probability = np.exp(-(local_test - best_result)/cooling_rate)
            if acceptance_probability > random.random():
                best_result = local_test
                best_point = coords

        center_point = best_point
        climb_points.append(best_point)
        climb_results.append(best_result)
        if len(convergence_points) == 0:
            convergence_points.append(best_point)
            convergence_results.append(best_result)
        elif convergence_results[len(convergence_points)-1] > best_result:
            convergence_points.append(best_point)
            convergence_results.append(best_result)
        else:
            convergence_results.append(convergence_results[len(convergence_results)-1])
            convergence_points.append(convergence_points[len(convergence_points) - 1])
        temp -= cooling_rate
    return convergence_results

def stat_analysis(averaging_count,test_function,std,dimension,population):
    iterations = int(10000 * dimension / population)
    sa_result = []
    sa_stats = {
        'min_val': 99999,
        'max_val': 0,
        'mean': 0,
        'median': 0,
        'std': 0,
        'convergence_total' : []
    }
    for i in range(averaging_count):
        result = simulated_annealing(100, 10, dimension, population, std, test_function)
        sa_result.append(result)
        if sa_stats.get('min_val') > result[len(result) - 1]:
            sa_stats['min_val'] = result[len(result) - 1]
        if sa_stats.get('max_val') < result[len(result) - 1]:
            sa_stats['max_val'] = result[len(result) - 1]
        sa_stats['convergence_total'] += result
    sa_stats['mean'] = np.mean(sa_stats.get('convergence_total'))
    sa_stats['median'] = np.median(sa_stats.get('convergence_total'))
    sa_stats['std'] = np.std(sa_stats.get('convergence_total'))

    sa_averaged_convergence = []
    for i in range(len(sa_result[0])):
        result_sum = 0
        for j in sa_result:
            result_sum += j[i]
        sa_averaged_convergence.append(result_sum / averaging_count)

    hc_result = []
    hc_stats = {
        'min_val': 99999,
        'max_val': 0,
        'mean': 0,
        'median': 0,
        'std': 0,
        'convergence_total' : []
    }
    for i in range(averaging_count):
        result = search(iterations, dimension, population, std, test_function, 'hc')
        hc_result.append(result)
        if hc_stats.get('min_val') > result[len(result) - 1]:
            hc_stats['min_val'] = result[len(result) - 1]
        if hc_stats.get('max_val') < result[len(result) - 1]:
            hc_stats['max_val'] = result[len(result) - 1]
        hc_stats['convergence_total'] += result
    hc_stats['mean'] = np.mean(hc_stats.get('convergence_total'))
    hc_stats['median'] = np.median(hc_stats.get('convergence_total'))
    hc_stats['std'] = np.std(hc_stats.get('convergence_total'))
    hc_averaged_convergence = []
    for i in range(len(hc_result[0])):
        result_sum = 0
        for j in hc_result:
            result_sum += j[i]
        hc_averaged_convergence.append(result_sum / averaging_count)

    ls_result = []
    ls_stats = {
        'min_val': 99999,
        'max_val': 0,
        'mean': 0,
        'median': 0,
        'std': 0,
        'convergence_total' : []
    }
    for i in range(averaging_count):
        result = search(iterations, dimension, population, std, test_function, 'ls')
        ls_result.append(result)
        if ls_stats.get('min_val') > result[len(result) - 1]:
            ls_stats['min_val'] = result[len(result) - 1]
        if ls_stats.get('max_val') < result[len(result) - 1]:
            ls_stats['max_val'] = result[len(result) - 1]
        ls_stats['convergence_total'] += result
    ls_stats['mean'] = np.mean(ls_stats.get('convergence_total'))
    ls_stats['median'] = np.median(ls_stats.get('convergence_total'))
    ls_stats['std'] = np.std(ls_stats.get('convergence_total'))
    ls_averaged_convergence = []
    for i in range(len(ls_result[0])):
        result_sum = 0
        for j in ls_result:
            result_sum += j[i]
        ls_averaged_convergence.append(result_sum / averaging_count)

    stat_table = [[np.round(sa_stats['min_val'],7),np.round(sa_stats['max_val'],7),np.round(sa_stats['mean'],7),np.round(sa_stats['median'],7),np.round(sa_stats['std'],7)],
                  [np.round(hc_stats['min_val'],7),np.round(hc_stats['max_val'],7),np.round(hc_stats['mean'],7),np.round(hc_stats['median'],7),np.round(hc_stats['std'],7)],
                  [np.round(ls_stats['min_val'],7),np.round(ls_stats['max_val'],7),np.round(ls_stats['mean'],7),np.round(ls_stats['median'],7),np.round(ls_stats['std'],7)]]
    stat_labelx = ['SA','HC','LS']
    stat_labely = ['Min','Max','Mean','Median','STD']
    fig,ax = plt.subplots(nrows=2)
    table = ax[0].table(cellText=stat_table,colLabels=stat_labely,rowLabels=stat_labelx,loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    ax[0].axis('off')
    ax[1].step(range(len(sa_averaged_convergence)), sa_averaged_convergence, label='simulated annealing')
    ax[1].step(range(len(hc_averaged_convergence)), hc_averaged_convergence, label='hill climber')
    ax[1].step(range(len(ls_averaged_convergence)), ls_averaged_convergence, label='local search')
    ax[1].set_yscale('log')
    formatter = FuncFormatter(lambda y, _: '{:.16g}'.format(y))
    ax[1].yaxis.set_major_formatter(formatter)
    # ax[1].yaxis.set_minor_formatter(formatter)
    ax[1].legend()
    fig.suptitle(f'Test_function: {test_function}, Dimensions: {dimension}, Population: {population}, {averaging_count} of averaged runs')
    plt.show()

if __name__=='__main__':
    # stat_analysis(averaging_count, test_function, std, dimension, population)
    averaging_count = 30
    test_function = 0
    std = 0.5
    dimension = 5
    population = 50
    sphere5 = multiprocessing.Process(target=stat_analysis,
                                      args=(averaging_count, test_function, std, dimension, population))
    sphere5.start()

    dimension = 10
    sphere10 = multiprocessing.Process(target=stat_analysis,
                                      args=(averaging_count, test_function, std, dimension, population))
    sphere10.start()

    dimension = 20
    sphere20 = multiprocessing.Process(target=stat_analysis,
                                      args=(averaging_count, test_function, std, dimension, population))
    sphere20.start()

    test_function = 1
    std = 20
    dimension = 5
    schwefel5 = multiprocessing.Process(target=stat_analysis,
                                      args=(averaging_count, test_function, std, dimension, population))
    schwefel5.start()

    dimension = 10
    schwefel10 = multiprocessing.Process(target=stat_analysis,
                                        args=(averaging_count, test_function, std, dimension, population))
    schwefel10.start()

    dimension = 20
    schwefel20 = multiprocessing.Process(target=stat_analysis,
                                        args=(averaging_count, test_function, std, dimension, population))
    schwefel20.start()

    test_function = 2
    std = 0.5
    dimension = 5
    dixonprice5 = multiprocessing.Process(target=stat_analysis,
                                        args=(averaging_count, test_function, std, dimension, population))
    dixonprice5.start()

    dimension = 10
    dixonprice10 = multiprocessing.Process(target=stat_analysis,
                                          args=(averaging_count, test_function, std, dimension, population))
    dixonprice10.start()

    dimension = 20
    dixonprice20 = multiprocessing.Process(target=stat_analysis,
                                          args=(averaging_count, test_function, std, dimension, population))
    dixonprice20.start()
