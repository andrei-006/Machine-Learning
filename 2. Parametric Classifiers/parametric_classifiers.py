# %% [markdown]
# #2: Parametric Classifiers

# %% [markdown]
# ## The Bayes classifier
# 
# 1. Classification using Gaussian distributions
# 2. Getting to know the data
# 3. Test sets
# 4. Univariate model
# 5. Probability density function
# 6. Posterior probabilities
# 7. Bayes classifier

# %% [markdown]
# ## 1. Classification using Gaussian distributions
# 
# We are starting with a very important notion in machine learning: probability distributions. Occurrences of data typically follow probability distributions that we know how to model.
# 
# Say, you want to classify apples vs. oranges. A _feature_ that you could use to classify them is their colour. We know, of course, that oranges are orange and apples (the golden delicious kind) are green, but each orange is a slightly different shade of orange. Likewise, the apples are all a different shade of green. If we would plot the colour values against the number of fruits with that colour, we would see, however, that there are probably more oranges with a certain type of shade than with other colours. They tend to follow known probability distributions.
# 
# In this assignment, we will assume data that has a normal distribution and try to estimate the **parameters** of the assumed normal distribution to correctly fit our data, hence the name **parametric classifiers**. We will then use Bayes' rule to build a classifier based on the probability distribution.
# 
# Just to refresh your mind, this is what a normal distribution looks like:
# ![Normal distribution for oranges](gaussian.png)

# %% [markdown]
# Instead of apples and oranges, we will try to classify flowers from Fisher's Iris dataset. The dataset contains the measurements of *length* and *width* of the *sepals* and *petals* of 150 flowers. 
# 
# (Petal-sepal.jpg)
# 
# Using the distribution of these 4 features (*length* and *width* of both *sepals* and *petals*), the flowers can then be classified as one of 3 species of Iris flower:
# 
# * Iris setosa
# * Iris versicolor
# * Iris virginica
# 
# This dataset is such a classic example that it is even included in machine learning libraries. The following code will load the dataset from `scikit-learn` (this was installed with conda) into the variable `iris`.
# 
# Run the code and inspect what data is contained in `iris`. Can you identify the 4 attributes? What other information is contained in `iris`?

# %%
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
iris

# %% [markdown]
# ## 2. Getting to know the data
# 
# The dataset is stored as a dictionary, a data structure in Python that resembles a Java(script) object. We can access items in the dictionary with a dot `.`, so we access the data and their target labels with `iris.data` and `iris.target`, these are both NumPy arrays. If we want to know what each digit means, we can access the names with `iris.target_names`.
# 
#  Run the code fragment and confirm what it is doing. Try to understand the indexing and print the following data:
# - The last five flowers. Expected result: an array with shape (5, 4).
# - Only the third feature of each flower. Expected result: an array with shape (150,).
# - The names of the first ten flowers. Expected result: an array with shape (10,).
# - Three separate arrays (one for each class). Expected result: three arrays with shape (50, 4). Try doing this without assuming anything about the indices for each class, i.e.: do not simply use `class1 = iris.data[:50, :]`. You can use `np.where`. This function takes in a boolean statement and returns the indices for which the statement is true. Example use: `np.where(iris.target == 0)` returns all indices where the target label is 0.  
# __Hint:__ Look at the indexing chapter in last week's NumPy lab for help.

# %%
print("Iris data: \n", iris.data)

print("Labels: \n", iris.target)

print("Labels (the class) ", iris.target_names)

last_five_flowers = iris.data[-5:, :]
third_feature_only = iris.data[:, 2]
first_ten_names = iris.target_names[iris.target[:10]]

setosa_flowers = iris.data[np.where(iris.target == 0)]

print("Setosa flowers: \n", setosa_flowers)

versicolor_flowers = iris.data[np.where(iris.target == 1)]

virginica_flowers = iris.data[np.where(iris.target == 2)]

# %% [markdown]
# Next, to get an idea of the distribution of our data, we can make plots.
# 
#  Run the following code to plot the petal length and width of each flower as a scatterplot

# %%
# From the Matplotlib library, import pyplot. We will refer to this library later as plt.
# This is a widely used library that lets you create images and plot your data.
from matplotlib import pyplot as plt

# Create a scatterplot of the first two features, and use their labels as colour values.
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
# Create a scatterplot of the third and fourth feature.
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

# %% [markdown]
#  How are the points distributed? Could you fit a probability distribution that you know on this data (e.g. uniform, normal, etc.)?
# 
# 
#  The points form three clusters, one for each species (colour). In the petal plot, these clusters are more clearly separated; in the sepal plot, they overlap more.
# You could fit a 2D normal (Gaussian) distribution to each class separately: most points lie near the centre of their cluster, with fewer farther away. One Gaussian for all three classes would poorly represent the separate clusters.

# %% [markdown]
# ## 3. Test sets
# 
# Now that we have an idea what our dataset looks like, our goal is to create a model that will predict the class of each flower based on its features. In order to evaluate how well the model fits, we will also need a separate test set where we can evaluate our final model on. For this, we will split the data randomly in a train and test set.
# 
#  Use the code below to split the dataset into a train and a test set.

# %%
from sklearn.model_selection import train_test_split #to split in train and test set

iris = datasets.load_iris()
# X is the feature vectors for the data points, and Y is the target (ground truth) class for those data points 
# the iris.data and iris.target entries are randomly divided into training and test sets.
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)

setosa_X_train = X_train[Y_train == 0]
versicolor_X_train = X_train[Y_train == 1]
virginica_X_train = X_train[Y_train == 2]

assert setosa_X_train.shape[0] != versicolor_X_train.shape[0]
assert setosa_X_train.shape[0] != virginica_X_train.shape[0]
assert versicolor_X_train.shape[0] != virginica_X_train.shape[0]

setosa_X_train.shape, versicolor_X_train.shape, virginica_X_train.shape

# %% [markdown]
# ## 4. Univariate model
# 
# Looking at the plots of the data from the previous section, you might assume that separating the different classes would be a lot easier based on the petal data (3rd and 4th variable) than on the sepal data (1st and 2nd variable), as it is easier to distinguish the different clusters in that plot. In fact, for now we will only focus on one variable, the petal length (3rd feature), as it looks like it might be useful just on its own and this will simplify the model a lot.

# %%
# We use the third feature
feature_idx = 2

# %% [markdown]
# Let's first take a look at the distribution of all flowers (both train and test) along this feature to confirm that our assumption of a normal distribution is correct. Take a look at the ditribution of the other features as well.

# %%
plt.hist(setosa_flowers[:,feature_idx], label=iris.target_names[0], alpha=0.7)
plt.hist(versicolor_flowers[:,feature_idx], label=iris.target_names[1], alpha=0.7)
plt.hist(virginica_flowers[:,feature_idx], label=iris.target_names[2], alpha=0.7)
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Number of flowers')
plt.legend()
plt.show()

# %% [markdown]
# That looks about correct! Now, let's find the parameters of the normal distribution that describe our data best. The parameters that we need to describe the distribution are the _mean_ and _standard deviation_.
# 
# Using the training data from each of 3 classes, compute the mean ($\mu$) and standard deviation ($\sigma$) for the *petal length* attribute. The Maximum Likelihood Estimators for these are given by
# 
# (4.1) $$\mu = \frac{\sum_{t=1}^Nx^t}{N}$$
# 
# (4.2) $$\sigma = \sqrt{\frac{\sum_{t=1}^N(x^t - m)^2}{N}}$$
# 
# __Hint:__ Try to use numpy's functions to perform operations on your input (e.g. `np.sum`, `np.sqrt`):

# %%
def compute_mean(x):
    mean = np.sum(x) / len(x)
    return mean
    
def compute_sd(x, mean):
    sd = np.sqrt(np.sum((x - mean) ** 2) / len(x))
    return sd

# Compute the mean for each flower type.
mean_setosa = compute_mean(setosa_X_train[:, feature_idx])
mean_versicolor = compute_mean(versicolor_X_train[:, feature_idx])
mean_virginica = compute_mean(virginica_X_train[:, feature_idx])

# Compute the standard deviation for each flower type.
sd_setosa = compute_sd(setosa_X_train[:, feature_idx], mean_setosa)
sd_versicolor = compute_sd(versicolor_X_train[:, feature_idx], mean_versicolor)
sd_virginica = compute_sd(virginica_X_train[:, feature_idx], mean_virginica)

# Print the computed means and standard deviations.
print("setosa", mean_setosa, sd_setosa)
print("versicolor", mean_versicolor, sd_versicolor)
print("virginica", mean_virginica, sd_virginica)

assert np.isclose(mean_setosa, 1.4729729729729728), "Expected a different mean"
assert np.isclose(mean_versicolor, 4.25), "Expected a different mean"
assert np.isclose(mean_virginica, 5.572222222222222), "Expected a different mean"

assert np.isclose(sd_setosa, 0.17652600857089654), "Expected a different standard deviation"
assert np.isclose(sd_versicolor, 0.44300112866673375), "Expected a different standard deviation"
assert np.isclose(sd_virginica, 0.547017728288333), "Expected a different standard deviation"

# %% [markdown]
#  The means match the centers of the three histograms, and the standard deviations also macth their spread

# %% [markdown]
# ## 5. Probability density function
# 
# The probability density function for a Gaussian distribution is defined as
# 
#  $$p(x|\mu, \sigma)=\frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$
# 
# That means that if we have estimates for $\mu$ and $\sigma$, we can compute the probability density for a specific value $x$.

# %%
from scipy.stats import norm

def normal_PDF(x, mean, sd):
    pdf = 0
    denominator = sd * np.sqrt(2 * np.pi) 
    numerator = np.exp(- (x - mean) ** 2 / (2 * sd ** 2))
    pdf = numerator / denominator
    return pdf

x = 0.5
mean = 2
sd = 0.5
my_pdf = normal_PDF(x, mean, sd)

scipy_pdf = norm.pdf(x, mean, sd)
print("Your pdf function outcome: ", my_pdf, " Scipy's function outcome: ", scipy_pdf)
assert np.isclose(my_pdf, scipy_pdf)

# And we plot the result of your PDF function for 100 points between 0 and 4: np.linspace(0, 4, 100)
xs = np.linspace(0, 4, 100)
plt.plot(xs, normal_PDF(xs, mean, sd))
plt.show()

# %% [markdown]
# We already made estimates for $\mu$ and $\sigma$ for the *petal length* for each of the 3 classes, so we can now also define PDFs for each separate class.
# 
#  Plot the 3 functions using [linspace](https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.linspace.html) for a range of x-values aside the histograms of the classes.

# %%
# Histograms of the flower types of the training set
plt.hist(setosa_X_train[:, feature_idx],
         label=iris.target_names[0], alpha=0.7, density=True)

plt.hist(versicolor_X_train[:, feature_idx],
         label=iris.target_names[1], alpha=0.7, density=True)

plt.hist(virginica_X_train[:, feature_idx],
         label=iris.target_names[2], alpha=0.7, density=True)


xs = np.linspace(0, 7, 1000)

plt.plot(xs, np.exp(-0.5 * ((xs - mean_setosa) / sd_setosa)**2) / (sd_setosa * np.sqrt(2 * np.pi)), label="setosa PDF")
plt.plot(xs, np.exp(-0.5 * ((xs - mean_versicolor) / sd_versicolor)**2) / (sd_versicolor * np.sqrt(2 * np.pi)), label="versicolor PDF")
plt.plot(xs, np.exp(-0.5 * ((xs - mean_virginica) / sd_virginica)**2) / (sd_virginica * np.sqrt(2 * np.pi)), label="virginica PDF")

plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel("Probability density")
plt.legend()
plt.show()

# %% [markdown]
# ## 6. Posterior probabilities
# 
# The plot above shows the probability densities for a feature $x$. For a normal distributed feature of a class $C_i$, you only need to know the mean and the standard deviation to be able to determine the probability of obtaining that data point $x$, i.e. $p(x | \mu_i, \sigma_i)$ or $p(x | C_i)$. So, $p(x | C_i)$ is the probability density of observing $x$ knowing that the datapoint comes from $C_i$.
# 
# - The prior probabillity is the probability density of a certain class $C_i$, without having any observations (knowledge); $p(C_i)$.
# - The posterior probabillity is the probability density of a certain class $C_i$, knowing a datapoint $x$ you observed; $p(C_i | x)$.
# 
# However, what would be useful for classification, is the posterior probabilities of the classes given the data, i.e. $P(C_i | x)$.
# 
# For a new test flower, we know its features, but not its species. The posterior probability \(P(\text{class}\mid\text{features})\) tells us how likely each species is given those measurements, so we can predict the species with the highest probability
# 
# To get the posterior probability, we can use Bayes' rule:
# 
# (6.1) $$P(C_i | x) =  \frac{p(x | C_i) P(C_i)}{p(x)} = \frac{p(x | C_i) P(C_i)}{\sum_{k=1}^K p(x | C_k) P(C_k)}$$
# 
# We will construct our classifier such that, after observing a datapoint $x$, we assign the point to the class $C_i$ with the higest $P(C_i | x)$, so to the class which is most likely.

# %%
def posterior(x, means, sds, priors, i):
    """
    Compute the posterior probability P(C_i | x).
    :param x: the sample to compute the posterior probability for.
    :param means: an array of means for each class.
    :param sds: an array of standard deviation values for each class.
    :param priors: an array of frequencies for each class.
    :param i: the index of the class to compute the posterior probability for.
    """
    posterior = 0
    pdf = normal_PDF(x, means[i], sds[i])
    numerator = pdf * priors[i]
    denominator = sum(normal_PDF(x, means[j], sds[j]) * priors[j] for j in range(len(means)))
    posterior = numerator / denominator
    return posterior

means = [mean_setosa, mean_versicolor, mean_virginica]
sds = [sd_setosa, sd_versicolor, sd_virginica]
priors = [
    setosa_X_train.shape[0]/X_train.shape[0],
    versicolor_X_train.shape[0]/X_train.shape[0],
    virginica_X_train.shape[0]/X_train.shape[0]
]

# Test out the code
flower_idx = 6
print("Flower belongs to class", iris.target_names[Y_train[flower_idx]])

# iterate over all classes
for i in range(3):
    x_post = posterior(X_train[flower_idx, feature_idx], means, sds, priors, i)
    print("Posterior probability for class", iris.target_names[i], ": ", x_post)

post_setosa = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 0)
post_versicolor = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 1)
post_virginica = posterior(X_train[flower_idx, feature_idx], means, sds, priors, 2)

assert np.isclose(post_setosa, 1.1048294835009998e-107, rtol = 0.0001, atol = 0.), "Expected a different posterior probability"
assert np.isclose(post_versicolor, 0.03817178391547811, rtol = 0.0001, atol = 0.), "Expected a different posterior probability"
assert np.isclose(post_virginica, 0.9618282160845218, rtol = 0.0001, atol = 0.), "Expected a different posterior probability"

# %% [markdown]
# Plot the posterior probabilities for all 3 classes. Does the plot of these 3 posteriors make sense based on the data?

# %%
xs = np.linspace(0, 7, 100)
means = [mean_setosa, mean_versicolor, mean_virginica]
sds = [sd_setosa, sd_versicolor, sd_virginica]
priors = [len(setosa_X_train) / len(X_train), len(versicolor_X_train) / len(X_train), len(virginica_X_train) / len(X_train)]

plt.plot(xs, posterior(xs, means, sds, priors, 0), label=iris.target_names[0])
plt.plot(xs, posterior(xs, means, sds, priors, 1), label=iris.target_names[1])
plt.plot(xs, posterior(xs, means, sds, priors, 2), label=iris.target_names[2])
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Posterior probability')
plt.legend()
plt.show()

# %% [markdown]
#  Where would you put the decision boundary for each class? In other words: where would you draw the line, separating each class. Could you formulate this mathematically?

# %% [markdown]
# ## 7. Bayes Classifier
# 
# Now that we can compute the posteriors for every class, constructing a classifier is easy. The Bayes classifier is defined as
# 
# - Classify as $C_i$ for which: $i = argmax_i\ P(C_i |x)$
# 
#  Write the code for the `classify` function. It should classify a single data point $x$ as one of the 3 classes, returning $0$, $1$ or $2$ based on the class the flower is most likely to belong to. The other arguments of the function should therefore be the vector of mean estimates `means` and the vector of standard deviation estimates `sds`, the class distribution `priors` and index `i` corresponds to class $C_i$.

# %%
def classify(x, means, sds, priors):
    classification = -1
    posteriors = posterior(x, means, sds, priors, 0), posterior(x, means, sds, priors, 1), posterior(x, means, sds, priors, 2)

    maxx = max(posteriors)

    if maxx == posteriors[0]:
        classification = 0
    elif maxx  == posteriors[1]:
        classification = 1
    elif maxx == posteriors[2]:
        classification = 2
    return classification

# Test out the code
flower_idxs = [5,20,30]
predicted_classes = np.zeros(3, dtype=np.int64)
for i, flower_idx in enumerate(flower_idxs):
    predicted_classes[i] = classify(X_train[flower_idx, feature_idx], means, sds, priors)

print("Predicted class", iris.target_names[predicted_classes])
print("Flower belongs to class", iris.target_names[Y_train[flower_idxs]])
assert (predicted_classes == Y_train[flower_idxs]).all()

# %%
def evaluate(X_test, Y_test, means, sds, priors):
    accuracy = 0
    for i in range(len(X_test)):
        prediction = classify(X_test[i], means, sds, priors)
        if prediction == Y_test[i]:
            accuracy += 1
    accuracy = accuracy / len(X_test)
    return accuracy

accuracy = evaluate(X_test[:, feature_idx], Y_test, means, sds, priors)

print(accuracy)
assert accuracy > 0.9, "Expected a higher accuracy"

# %% [markdown]
# Let's return to our scatterplots and see how your classifier makes decisions. For this, we also plot the decision boundary. The function to create the decision boundaries does this in a very simple way:

# %%
def decision_boundary(means, sds, priors):
    decision_boundaries = []
    for i in range(len(means)):
        for j in range(i + 1, len(means)):
            A = 1 / (2 * sds[j]**2) - 1 / (2 * sds[i]**2)
            B = means[i] / sds[i]**2 - means[j] / sds[j]**2
            C = means[j]**2 / (2 * sds[j]**2) - means[i]**2 / (2 * sds[i]**2)
            C += np.log((priors[i] * sds[j]) / (priors[j] * sds[i]))
            roots = []
            if abs(A) < 1e-12:
                if abs(B) > 1e-12:
                    roots.append(-C / B)
            else:
                delta = B * B - 4 * A * C
                if delta >= 0:
                    roots.append((-B - np.sqrt(delta)) / (2 * A))
                    roots.append((-B + np.sqrt(delta)) / (2 * A))
            for x in roots:
                if 1 <= x <= 7:
                    probabilities = []
                    for k in range(len(means)):
                        probabilities.append(posterior(x, means, sds, priors, k))
                    if probabilities[i] >= max(probabilities) - 1e-8:
                        if not any(abs(x - boundary) < 1e-8 for boundary in decision_boundaries):
                            decision_boundaries.append(x)
    decision_boundaries.sort()
    return decision_boundaries

# Create a scatterplot of the third and fourth feature.
feature_idx2 = 3

plt.scatter(iris.data[:, feature_idx], iris.data[:, feature_idx2], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel(iris.feature_names[feature_idx2])
decision_boundaries = decision_boundary(means, sds, priors)
for boundary in decision_boundaries:
    plt.axvline(x=boundary)

plt.show()


