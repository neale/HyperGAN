from __future__ import division
import numpy as np
from abc import abstractmethod
import logging
import warnings

from .base import Attack
from .base import call_decorator
from .. import distances
from ..utils import crossentropy


""" Main attack that we will use """
# sample new network every time, 

class HyperIterativeProjectedGradientBaseAttack(Attack):
    """Base class for iterative (projected) gradient attacks.

    Concrete subclasses should implement __call__, _gradient
    and _clip_perturbation.

    TODO: add support for other loss-functions, e.g. the CW loss function,
    see https://github.com/MadryLab/mnist_challenge/blob/master/pgd_attack.py
    """

    @abstractmethod
    def _gradient(self, a, x, class_, strict=True):
        raise NotImplementedError

    @abstractmethod
    def _clip_perturbation(self, a, noise, epsilon):
        raise NotImplementedError

    @abstractmethod
    def _check_distance(self, a):
        raise NotImplementedError
    
    """ unchanged """
    def _get_mode_and_class(self, a):
        # determine if the attack is targeted or not
        target_class = a.target_class()
        targeted = target_class is not None

        if targeted:
            class_ = target_class
        else:
            class_ = a.original_class
        return targeted, class_
    
    """ here is where we sample a new network """
    """ requires a foolbox instance of the hyperGAN """
    def _run(self, a, binary_search,
             epsilon, stepsize, iterations,
             random_start, return_early):
        if not a.has_gradient():
            warnings.warn('applied gradient-based attack to model that'
                          ' does not provide gradients')
            return

        self._check_distance(a)

        targeted, class_ = self._get_mode_and_class(a)

        if binary_search:
            if isinstance(binary_search, bool):
                k = 20
            else:
                k = int(binary_search)
            return self._run_binary_search(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early, k=k)
        else:
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

    """ unused, unchanged """
    def _run_binary_search(self, a, epsilon, stepsize, iterations,
                           random_start, targeted, class_, return_early, k):

        factor = stepsize / epsilon

        def try_epsilon(epsilon):
            stepsize = factor * epsilon
            return self._run_one(
                a, epsilon, stepsize, iterations,
                random_start, targeted, class_, return_early)

        for i in range(k):
            if try_epsilon(epsilon):
                logging.info('successful for eps = {}'.format(epsilon))
                break
            logging.info('not successful for eps = {}'.format(epsilon))
            epsilon = epsilon * 1.5
        else:
            logging.warning('exponential search failed')
            return

        bad = 0
        good = epsilon

        for i in range(k):
            epsilon = (good + bad) / 2
            if try_epsilon(epsilon):
                good = epsilon
                logging.info('successful for eps = {}'.format(epsilon))
            else:
                bad = epsilon
                logging.info('not successful for eps = {}'.format(epsilon))

    """ main iterative method to use (no binary search) """
    def _run_one(self, a, epsilon, stepsize, iterations,
                 random_start, targeted, class_, return_early):
        min_, max_ = a.bounds()
        s = max_ - min_

        original = a.original_image.copy()

        if random_start:
            # using uniform noise even if the perturbation clipping uses
            # a different norm because cleverhans does it the same way
            """ lol """
            noise = np.random.uniform(
                -epsilon * s, epsilon * s, original.shape).astype(
                    original.dtype)
            x = original + self._clip_perturbation(a, noise, epsilon)
            strict = False  # because we don't enforce the bounds here
        else:
            x = original
            strict = True

        success = False
        for _ in range(iterations):
            # get net network here
            a._new_model()

            gradient = self._gradient(a, x, class_, strict=strict)
            # non-strict only for the first call and
            # only if random_start is True
            strict = True
            if targeted:
                gradient = -gradient

            # untargeted: gradient ascent on cross-entropy to original class
            # targeted: gradient descent on cross-entropy to target class
            x = x + stepsize * gradient

            x = original + self._clip_perturbation(a, x - original, epsilon)

            x = np.clip(x, min_, max_)

            logits, is_adversarial = a.predictions(x)
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                if targeted:
                    ce = crossentropy(a.original_class, logits)
                    logging.debug('crossentropy to {} is {}'.format(
                        a.original_class, ce))
                ce = crossentropy(class_, logits)
                logging.debug('crossentropy to {} is {}'.format(class_, ce))
            if is_adversarial:
                if return_early:
                    return True
                else:
                    success = True
        return success


class HyperLinfinityGradientMixin(object):
    def _gradient(self, a, x, class_, strict=True):
        gradient = a.gradient(x, class_, strict=strict)
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient


class HyperLinfinityClippingMixin(object):
    def _clip_perturbation(self, a, perturbation, epsilon):
        min_, max_ = a.bounds()
        s = max_ - min_
        clipped = np.clip(perturbation, -epsilon * s, epsilon * s)
        return clipped


class HyperLinfinityDistanceCheckMixin(object):
    def _check_distance(self, a):
        if not isinstance(a.distance, distances.Linfinity):
            """ Annoying """
            #logging.warning('Running an attack that tries to minimize the'
            #                ' Linfinity norm of the perturbation without'
            #                ' specifying foolbox.distances.Linfinity as'
            #                ' the distance metric might lead to suboptimal'
            #                ' results.')


class HyperLinfinityBasicIterativeAttack(
        HyperLinfinityGradientMixin,
        HyperLinfinityClippingMixin,
        HyperLinfinityDistanceCheckMixin,
        HyperIterativeProjectedGradientBaseAttack):

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.05,
                 iterations=10,
                 random_start=False,
                 return_early=True):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


HyperBasicIterativeMethod = HyperLinfinityBasicIterativeAttack
HyperBIM = HyperBasicIterativeMethod


class HyperProjectedGradientDescentAttack(
        HyperLinfinityGradientMixin,
        HyperLinfinityClippingMixin,
        HyperLinfinityDistanceCheckMixin,
        HyperIterativeProjectedGradientBaseAttack):

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.01,
                 iterations=40,
                 random_start=False,
                 return_early=True):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)


class HyperMomentumIterativeAttack(
        HyperLinfinityClippingMixin,
        HyperLinfinityDistanceCheckMixin,
        HyperIterativeProjectedGradientBaseAttack):

    def _gradient(self, a, x, class_, strict=True):
        # get current gradient
        gradient = a.gradient(x, class_, strict=strict)
        gradient = gradient / max(1e-12, np.mean(np.abs(gradient)))

        # combine with history of gradient as new history
        self._momentum_history = \
            self._decay_factor * self._momentum_history + gradient

        # use history
        gradient = self._momentum_history
        gradient = np.sign(gradient)
        min_, max_ = a.bounds()
        gradient = (max_ - min_) * gradient
        return gradient

    def _run_one(self, *args, **kwargs):
        # reset momentum history every time we restart
        # gradient descent
        self._momentum_history = 0
        return super(MomentumIterativeAttack, self)._run_one(*args, **kwargs)

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 binary_search=True,
                 epsilon=0.3,
                 stepsize=0.06,
                 iterations=10,
                 decay_factor=1.0,
                 random_start=False,
                 return_early=True):

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        assert epsilon > 0

        self._decay_factor = decay_factor

        self._run(a, binary_search,
                  epsilon, stepsize, iterations,
                  random_start, return_early)

HyperMomentumIterativeMethod = HyperMomentumIterativeAttack
