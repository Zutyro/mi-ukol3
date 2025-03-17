
import numpy as np
import matplotlib.pyplot as plt
import random

def test_domain(point_values, domain):
    for x in point_values:
        if x < domain[0] or x > domain[1]:
            return False
    return True

def sphere(point_values):
    sum1 = 0
    for x in point_values:
        sum1 = sum1 + (x ** 2)
    y = sum1
    return y


def schwefel(point_values):
    sum1 = 0
    for x in point_values:
        sum1 = sum1 + x * np.sin(np.sqrt(np.abs(x)))
    y = 418.9829 * len(point_values) - sum1
    return y

def dixonprice(point_values):
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



# def search(max_iterations: int, dimensions: int, population: int, standard_deviation: float, function_type: int, search_alg='hc'):
#     if search_alg != 'hc' and search_alg != 'ls':
#         raise ValueError('search_alg can only be hc or ls')
#     domain = []
#     match function_type:
#         case 0:
#             function_name = 'sphere'
#             test_function = sphere
#             domain = [-5.12,5.12]
#         case 1:
#             function_name ='trid'
#             test_function = trid
#             domain = [-dimensions**2,dimensions**2]
#         case 2:
#             function_name ='schwefel'
#             test_function = schwefel
#             domain = [-500,500]
#         case 3:
#             function_name ='dixonprice'
#             test_function = dixonprice
#             domain = [-10,10]
#         case 4:
#             function_name ='rosenbrock'
#             test_function = rosenbrock
#             domain = [-2.048,2.048]
#         case _:
#             raise ValueError('Function_type outside of range. Accepts values between 0 and 4')
#     center_point = [random.uniform(domain[0], domain[1]) for i in range(dimensions)]
#     center_result = test_function(center_point)
#     best_result = 0
#     best_point = 0
#     convergence_points = []
#     convergence_results = []
#     climb_results = []
#     climb_points = []
#     for i in range(max_iterations):
#         best_result = 0
#         best_point = 0
#         for x in range(population):
#             coords = generate_coords(domain, center_point, standard_deviation, dimensions)
#             local_test = test_function(coords)
#             if search_alg == 'hc' and best_result == 0:
#                 best_result = local_test
#                 best_point = coords
#             if search_alg == 'ls' and best_result == 0:
#                 best_result = center_result
#                 best_point = center_point
#             if local_test < best_result:
#                 best_result = local_test
#                 best_point = coords
#         center_point = best_point
#         climb_points.append(best_point)
#         climb_results.append(best_result)
#         if len(convergence_points) == 0:
#             convergence_points.append(best_point)
#             convergence_results.append(best_result)
#         if convergence_results[len(convergence_points)-1] > best_result:
#             convergence_points.append(best_point)
#             convergence_results.append(best_result)
#     fig, ax = 0,0
#     if dimensions == 2:
#         fig, ax = plt.subplots(3)
#         climb_points = np.array(climb_points).T
#         xx,yy = np.meshgrid(np.linspace(domain[0],domain[1],500), np.linspace(domain[0],domain[1],500))
#         zz = test_function((xx,yy))
#
#         ax[2].pcolor(xx, yy, zz)
#         ax[2].scatter(climb_points[0],climb_points[1],color='Red')
#     else:
#         fig, ax = plt.subplots(2)
#     step = ax[1].step(convergence_results,range(len(convergence_results)))
#     fig.set_figheight(8)
#     fig.set_figwidth(8)
#     data = [['Minimum:',convergence_results[len(convergence_results)-1]],['Parametry:', convergence_points[len(convergence_points)-1]]]
#     table = ax[0].table(cellText=data, colWidths = [0.15, 0.25], loc='center')
#     table.set_fontsize(14)
#     table.scale(3, 3)
#     ax[0].axis('off')
#     fig.suptitle(f'Search alg:{search_alg.upper()}\nDimensions:{dimensions}, Population:{population}, STD:{standard_deviation},\nTest function:{function_name}')
#     plt.show()
#     print(f'Best result:{convergence_results[len(convergence_results)-1]} in {convergence_points[len(convergence_points)-1]}')


def simulated_annealing(temp0: int,tempF: int, dimensions: int, population: int, standard_deviation: float, function_type: int):
    domain = []
    match function_type:
        case 0:
            function_name = 'sphere'
            test_function = sphere
            domain = [-5.12, 5.12]
        case 1:
            function_name = 'dixonprice'
            test_function = dixonprice
            domain = [-10, 10]
        case 2:
            function_name = 'schwefel'
            test_function = schwefel
            domain = [-500, 500]
        case _:
            raise ValueError('Function_type outside of range. Accepts values between 0 and 2')



    center_point = [random.uniform(domain[0], domain[1]) for i in range(dimensions)]
    center_result = test_function(center_point)
    best_result = 0
    best_point = 0
    convergence_points = []
    convergence_results = []
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
        if convergence_results[len(convergence_points)-1] > best_result:
            convergence_points.append(best_point)
            convergence_results.append(best_result)
        temp -= cooling_rate


if __name__=='__main__':
    # search(20, 2, 5, 1, 0, 'hc')
    simulated_annealing(100,10,2,3,2,0)
