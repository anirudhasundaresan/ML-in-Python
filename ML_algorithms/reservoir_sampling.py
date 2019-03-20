# https://gregable.com/2007/10/reservoir-sampling.html
# https://www.quora.com/What-is-an-intuitive-explanation-of-reservoir-sampling
# very nice video with proofs - https://www.youtube.com/watch?v=Ybra0uGEkpM


# To retrieve k random numbers from an array of undetermined size we use a technique called reservoir sampling.
# do both normal and weighted sampling

# We do not know the full size of the stream N
# Goal is to sample a set k from this stream. Probability that each of these elements belonging to this set must be equally likely.

# https://stackoverflow.com/questions/2612648/reservoir-sampling
# http://data-analytics-tools.blogspot.com/2009/09/reservoir-sampling-algorithm-in-perl.html

import random
import numpy as np

def sample(iterable, k):
    reservoir = []
    for ind, item in enumerate(iterable):
        if ind < k:
            reservoir.append(item)
        else:
            m = random.randint(0, ind)
            # you can have this comparative condition either with m and then separately --> replace = random.randint(0,len(sample)-1); sample[replace] = line
            if m < k:
                reservoir[m] = item
    return reservoir

iterable = range(200)
np.random.shuffle(list(iterable))
print(sample(iterable, 5))

#  more about this: https://www.youtube.com/watch?v=s_Za9GlD0ek
