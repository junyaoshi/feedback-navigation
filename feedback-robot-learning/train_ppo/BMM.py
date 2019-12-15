import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os


def train_bmm_model(loss, a, update, base_path, correct, feedback_acc):
    # outliers detection
    max_perc = np.percentile(loss, 97)
    min_perc = np.percentile(loss, 3)
    correct = correct[(loss<=max_perc) & (loss>=min_perc)]
    loss = loss[(loss<=max_perc) & (loss>=min_perc)]

    bmm_model_maxLoss = max_perc
    bmm_model_minLoss = min_perc + 10e-6

    loss = (loss - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)

    loss[loss >= 1] = 1-1e-3
    loss[loss <= 0] = 1e-3

    bmm_model = BetaMixture1D(max_iters=5, max_loss=bmm_model_maxLoss, min_loss=bmm_model_minLoss)
    bmm_model.fit(loss)

    bmm_model.create_lookup(1)
    if update % 20 == 0:
        plot_bmm_fitting(loss, bmm_model, a, update, base_path, correct, feedback_acc)

    return bmm_model

def plot_bmm_fitting(loss, bmm_model, a, update, base_path, correct, feedback_acc):
    plt.figure()
    correct_loss = loss[correct == True]
    wrong_loss = loss[correct == False]
    combined_loss = [correct_loss, wrong_loss]
    n, bins, _ = plt.hist(combined_loss, 50, stacked=True, density=True, alpha=0.7, color=["blue", "red"])
    plt.plot(bmm_model.x_l, bmm_model.weighted_likelihood(bmm_model.x_l, 0), label="bmm_pdf_0")
    plt.plot(bmm_model.x_l, bmm_model.weighted_likelihood(bmm_model.x_l, 1), label="bmm_pdf_1")
    plt.plot(bmm_model.x_l, bmm_model.lookup, label="bmm_posterior")
    cum = 0
    n = np.mean(n, axis=0)
    for i in range(len(n)):
        cum += n[i]
        if cum > feedback_acc * np.sum(n):
            break
    plt.vlines(bins[i + 1], 0, 100)
    plt.legend()
    plt.xlabel('Loss')
    plt.ylabel('Probability')
    plt.ylim([0, np.max(n) * 1.1])
    plot_dir = os.path.join(base_path, "bmm_fitting_plot", "action_{}".format(a))
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plt.savefig(os.path.join(plot_dir, "update_{}_NLoss_{}.jpg".format(update, len(loss))))
    plt.close("all")

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[2, 6],
                 betas_init=[6, 2],
                 weights_init=[0.5, 0.5],
                 max_loss=1, min_loss=0):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-6
        self.max_loss = max_loss
        self.min_loss = min_loss

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 5e-3
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        self.x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(self.x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t

    def look_lookup(self, loss):
        loss = (loss - self.min_loss) / (self.max_loss - self.min_loss + 1e-6)
        loss[loss >= 1] = 1-1e-3
        loss[loss <= 0] = 1e-3
        x = loss
        x_i = np.array((self.lookup_resolution * x).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i >= self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)