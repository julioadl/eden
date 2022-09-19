import matplotlib.pyplot as plt
import numpy as np
import os

K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(trial_num):

    # Load dataset
    train_path = os.path.join('.', 'train.csv')
    x_all, z_all = load_gmm_dataset(train_path)

    # Split into labeled and unlabeled examples
    labeled_idxs = (z_all != UNLABELED).squeeze()
    x_tilde = x_all[labeled_idxs, :]   # Labeled examples
    z_tilde = z_all[labeled_idxs, :]   # Corresponding labels
    x = x_all[~labeled_idxs, :]        # Unlabeled examples

    # Initialize mu and sigma by splitting the n_examples data point uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    n = x.shape[0]
    mu = []
    sigma = []
    len_x = x.shape[0]
    size_sample = int(len_x / K)
    for j in range(K):
        sample_j = np.random.choice(len_x, size_sample)
        mu_j = np.mean(x[sample_j,:], axis=0)
        s_j = np.dot((x[sample_j,:] - mu_j).T, (x[sample_j,:] - mu_j)) / size_sample
        mu.append(mu_j)
        sigma.append(s_j)

    # Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.array([1 / K] * K)

    # Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (n, K)
    w = np.array([phi for j in range(len_x)])
    w = run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma)

    # Plot  predictions
    z_pred = np.zeros(n)
    if w is not None: 
        for i in range(n):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)

def run_semi_supervised_em(x, x_tilde, z_tilde, w, phi, mu, sigma, max_iter=1000):

    """
    Args:
        x: Design matrix of unlabeled examples of shape (n_examples_unobs, dim).
        x_tilde: Design matrix of labeled examples of shape (n_examples_obs, dim).
        z_tilde: Array of labels of shape (n_examples_obs, 1).
        w: Initial weight matrix of shape (n_examples, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (dim,).
        sigma: Initial cluster covariances, list of k arrays of shape (dim, dim)
        max_iter: Max iterations. No need to change this

    Returns:
        Updated weight matrix of shape (n_examples, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    it = 0
    ll = prev_ll = None
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        prev_ll = ll

        n = w.shape[0]
        n_tilde, dims_tilde = x_tilde.shape
        # (1) E-step: Update your estimates in w

        w_tilde = np.zeros((n_tilde, K))
        for j in range(K):
            mu_j = mu[j]
            sigma_j = sigma[j]
            phi_j = phi[j]
            w[:,j] = normal_distribution(x, mu_j, sigma_j) * phi_j
            w_tilde[:, j] = (z_tilde == j).squeeze()


        w = w / w.sum(axis=1, keepdims=True) #Normalized by the number of latent classes

        phi = (np.sum(w, axis=0) + alpha * np.sum(w_tilde, axis=0))/ (n + alpha * n_tilde)

        for j in range(K):
            w_j = w[:,j]#np.expand_dims(w[:,j], -1)
            w_tilde_j = w_tilde[:, j]
            mu[j] = (np.dot(w_j, x) + alpha * np.dot(w_tilde_j, x_tilde)) / (np.sum(w_j) + alpha * np.sum(w_tilde_j))
            sigma[j] = ((x - mu[j]).T.dot(np.diag(w_j)).dot(x - mu[j]) + alpha * (x_tilde - mu[j]).T.dot(np.diag(w_tilde_j)).dot(x_tilde - mu[j])) / (np.sum(w_j) + alpha * np.sum(w_tilde_j))

        p_x = np.zeros(n)
        for j in range(K):
            p_x_mid_z = normal_distribution(x, mu[j], sigma[j])
            p_x += p_x_mid_z * phi[j]

        p_x_z = np.zeros(n_tilde)
        for j in range(K):
            p_x_z += normal_distribution(x_tilde, mu[j], sigma[j]) * phi[j]

        ll = np.sum(np.log(p_x)) + alpha * np.sum(np.log(p_x_z))
        it += 1
        print(f'Iteration {it} with likelihood {ll}')

    return w

def normal_distribution(x, mu, sigma):
    n = x.shape[0]
    dims = x.shape[1]
    x = np.expand_dims(x, -1)
    mu = np.expand_dims(mu, 0)
    constant = 1 / (np.power(2 * np.pi, dims / 2) * np.sqrt(np.linalg.det(sigma)))
np.linalg.pinv(sigma)).shape}')
    cov = - 0.5 * np.matmul(np.matmul((x - mu.T).transpose(0,2,1), np.linalg.pinv(sigma)),(x - mu.T))
    cov = np.squeeze(cov)
    return constant * np.exp(cov)


def plot_gmm_preds(x, z, with_supervision, plot_id):

    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('.', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model.

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (n_examples, dim)
        z: NumPy array shape (n_exampls, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    for t in range(NUM_TRIALS):
        main(trial_num=t)
