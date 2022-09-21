class Bayes:
    """Bayes class required from coding exercise 1. The functionality is what is
    requested, and is done to the best of our understanding."""

    def __init__(self, hypothesis: list, priors: list, observations: list, likelihood_array: list):
        """Method to initialize the internal bayes system.

        Method assumes that the given hypothesis and observations list match up one to one to the likelihood array
        in the same order, with the hypothesis elements being the first dimension, and the observations representing the
        second dimension.

        :param hypothesis: List of the hypothesis
        :param priors: List of priors, meaning the probability of the hypothesis holding. Assumed to be the same length
            as hypothesis.
        :param observations: List of possible observations for each of the given hypothesis
        :param likelihood_array: Probabilities for each likelihood
        """

        self.hyp = hypothesis
        self.priors = priors
        self.obs = observations
        self.like = likelihood_array

    def likelihood(self, observation, hypothesis) -> float:
        # Finding the indexes.
        loc_obs = self.obs.index(observation)
        loc_hyp = self.hyp.index(hypothesis)

        return self.like[loc_hyp][loc_obs]

    def norm_constant(self, observation) -> float:
        loc_obs = self.obs.index(observation)
        prob_obs_per_hypothesis = [elem[loc_obs] for elem in self.like]

        # Now we just need to do element wise operations. Google is our friend and is always helpful
        temp = [a * b for a, b in zip(prob_obs_per_hypothesis, self.priors)]

        return sum(temp)

    def single_posterior_update(self, observation, priors) -> list:
        # This should return an array, as we're calculating the posterior for all the given priors
        # At least, that's how I can make sense of this?

        # So, first step is calculating the normalizing constant, which is used no matter what
        norm_const = self.norm_constant(observation)

        # Now, for each hypothesis, I'll store them in results afterwards
        results = []

        for i in range(len(self.hyp)):
            elem = self.hyp[i]
            prior_value = priors[i]

            likelihood_val = self.likelihood(observation, elem)

            posterior_value = (likelihood_val*prior_value)/norm_const
            results.append(posterior_value)

        return results

    def compute_posterior(self, observations: list) -> list:
        """Method to calculate posterior when receiving multiple items.

        Method assumes that the initial priors are the valid ones.

        Assumes each observation is individual. IE, the probability of observation i is multiplied
        with the probability of observation i+1 directly, ignoring the previous value

        This last assumption might be very wrong"""

        # So, for each hypothesis, the results gets updated at each individual step
        # because that is easy and efficient
        results = [1] * len(self.hyp)
        for elem in observations:
            temp_res = self.single_posterior_update(elem, self.priors)

            # Range of temp_res should be the same as the range of results
            # This just updates the values inside manually
            for i in range(len(temp_res)):
                results[i] = temp_res[i] * results[i]

            # Now I normalize the priors, and then save the new values inside
            sum_res_temp = sum(results)
            for i in range(len(temp_res)):
                results[i] = results[i]/sum_res_temp

        return results


if __name__ == '__main__':
    # This was just copied and pasted from the given code, because that is lazy and easy and good!
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likehood of Bowl1 and vanilla, or 35/50
    likelihood = [[15 / 50, 35 / 50], [30 / 50, 20 / 50]]
    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)

    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    p_2 = b.compute_posterior(["chocolate", "vanilla"])
    print("chocolate, vanilla - posterior: %s" % p_2)

    # Here is the code for the archery problem exercises
    hypos_arch = ['Beginner', 'Intermediate', 'Advanced', 'Expert']
    priors_arch = [0.25, 0.25, 0.25, 0.25]
    obs_arch = ['yellow', 'red', 'blue', 'black', 'white']
    likelihood_arch = [[0.05, 0.1, 0.4, 0.25, 0.2],
                       [0.1, 0.2, 0.4, 0.2, 0.1],
                       [0.2, 0.4, 0.25, 0.1, 0.05],
                       [0.3, 0.5, 0.125, 0.05, 0.025]]

    b_arch = Bayes(hypos_arch, priors_arch, obs_arch, likelihood_arch)

    # This is what the exercise says that we saw. So the sequence of shots taken
    observe_test = ['yellow', 'white', 'blue', 'red', 'red', 'blue']
    prob_posterior_arch = b_arch.compute_posterior(observe_test)

    # I'm not sure if this should be normalized afterwards... as these won't necessarily add up to 1
    # For now, I'll just assume they don't have to be normalized, because I'm lazy
    prob_intermediate = prob_posterior_arch[1]
    print('Probability of intermediate level archer: %s' %prob_intermediate)
    print('The probability of all archers is: ' + str(prob_posterior_arch))

    max_index = prob_posterior_arch.index(max(prob_posterior_arch))
    most_likely_arch = hypos_arch[max_index]
    print('Most likely archer is ' + most_likely_arch)
    print('With a probability of: ' + str(max(prob_posterior_arch)))

    # This is just writing everything into a text file
    f = open('answers.txt', 'w')
    # f.write(str(l) + '\n')
    # f.write(str(n_c) + '\n')
    # I grabbed the 0th element because we only care about the probabilities of it being from
    # bowl 1
    f.write(str(p_1[0]) + '\n')
    # Same idea as previous, but for bowl 2
    f.write(str(p_2[1]) + '\n')
    f.write(str(prob_intermediate) + '\n')
    f.write(str(most_likely_arch) + '\n')

    f.close()
