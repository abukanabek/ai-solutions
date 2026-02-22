import numpy as np
from sklearn.linear_model import LogisticRegression

def train_model(X, y):
    model = LogisticRegression()
    model.fit(X, y)
    return model

def main():
    mode = ""
    input_file, output_file = 'input.txt', 'output.txt'
    if mode == 'LOCAL':
        input_file, output_file = 'NIPS_input.txt', 'NIPS_output.txt'
    with open(input_file) as f:
        _, N_test, _, _ = map(int, f.readline().split())

    with open(input_file) as f:
        N_train, N_test, D, N_labeled = map(int, f.readline().split()) 
        f.readline() # blank line
        train_latents = np.zeros((N_train, D)) 
        for i in range(N_train): 
            train_latents[i] = list(map(float, f.readline().split()))
        f.readline() # blank line 

        test_latents = np.zeros((N_test, D))
        for i in range(N_test): 
            test_latents[i] = list(map(float, f.readline().split()))
        f.readline() # blank line 


        labeled_indices = np.zeros(N_labeled, dtype=np.int64) 
        labeled_labels = np.zeros(N_labeled, dtype=np.int64) 
        for i in range(N_labeled): 
            parts = f.readline().split() 
            labeled_indices[i] = int(parts[0]) 
            labeled_labels[i] = int(parts[1])

    X = train_latents[labeled_indices]
    y = labeled_labels

    model = train_model(X, y)

    y_pred = model.predict(test_latents)

    with open(output_file, "w") as f:
        for i in range(N_test):
            f.write(f"{y_pred[i]}\n")


if __name__ == "__main__":
    main()
