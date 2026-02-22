import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.decomposition import PCA


def generate_corgi():
    dogs = []
    for _ in range(1000):
        a, b = np.random.uniform(), np.random.uniform()
        X = a * np.cos(2*np.pi * b)
        Y = a * np.sin(2*np.pi * b)
        dogs.append((X, Y))
    return np.array(dogs)

def generate_pro():
    dogs = []
    for _ in range(1000):
        X, Y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        while X**2 + Y**2 > 1:
            X, Y = np.random.uniform(-1, 1), np.random.uniform(-1, 1)
        dogs.append((X, Y))
    return np.array(dogs)

def main():

    mode = ""
    input_file, output_file = 'input.txt', 'output.txt'
    if mode == 'LOCAL':
        input_file, output_file = 'CORGI_input.txt', 'CORGI_output.txt'

    populations = []
    with open(input_file) as f:
        N_populations, N_dogs = map(int, f.readline().split())
        _ = f.readline()
        for _ in range(N_populations):
            dogs = []
            for __ in range(N_dogs):
                X, Y = map(float, f.readline().split())
                dogs.append((X, Y))
            populations.append(dogs)
        populations = np.array(populations)
        
    b = 10
    bins = np.zeros((populations.shape[0], b+1))

    for i, pop in enumerate(populations):
        scores = sorted([round(X.item()**2+Y.item()**2, 3) for X, Y in pop])
        cur = 0
        for sc in scores:
            if sc > 1/b*(cur+1):
                cur += 1
            bins[i][cur] += 1

        for j in range(b):
            bins[i][b] += bins[i][j] * (j+1)/b
    
    y_pred = (bins[:, -1] > np.quantile(bins[:, -1], 0.5)).astype(int)

    with open(output_file, "w") as f:
        for i in range(N_populations):
            f.write(f"{y_pred[i]}\n")


if __name__ == "__main__":
    main()
