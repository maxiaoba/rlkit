# from https://github.com/deepmind/open_spiel
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.linalg as la

from open_spiel.python.egt import utils
from open_spiel.python.egt.alpharank import *
import matplotlib.patches as patches

import matplotlib
matplotlib.rcParams.update({'font.family': 'serif'})
import matplotlib.pyplot as plt

from rlkit.util.alpharank_visualizer import plot_pi_vs_alpha, NetworkPlot

def sweep_pi_vs_alpha(payoff_tables,
                      strat_labels=None,
                      warm_start_alpha=None,
                      m=50,
                      rtol=1e-5,
                      atol=1e-8):
  """Computes stationary distribution, pi, for range of selection intensities.

  The range of selection intensities is defined in alpha_list and corresponds
  to the temperature of the Fermi selection function.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a numpy array, or a _PayoffTableInterface
      object.
    strat_labels: Human-readable strategy labels. See get_strat_profile_labels()
      in utils.py for formatting details.
    warm_start_alpha: Initial value of alpha to use.
    visualize: Plot the sweep results.
    return_alpha: Whether to return the final alpha used.
    m: AlphaRank population size.
    rtol: The relative tolerance parameter for np.allclose calls.
    atol: The absolute tolerance parameter for np.allclose calls.

  Returns:
   pi: AlphaRank stationary distribution.
   alpha: The AlphaRank selection-intensity level resulting from sweep.
  """

  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  num_populations = len(payoff_tables)
  num_strats_per_population =\
    utils.get_num_strats_per_population(payoff_tables, payoffs_are_hpt_format)

  if num_populations == 1:
    num_profiles = num_strats_per_population[0]
  else:
    num_profiles = utils.get_num_profiles(num_strats_per_population)

  assert strat_labels is None or isinstance(strat_labels, dict)\
      or (len(strat_labels) == num_profiles)

  pi_list = np.empty((num_profiles, 0))
  alpha_list = []
  num_iters = 0
  alpha_mult_factor = 2.

  if warm_start_alpha is not None:
    alpha = warm_start_alpha
    alpharank_succeeded_once = False
  else:
    alpha = 1e-4  # Reasonable default for most games, can be user-overridden

  while 1:
    try:
      _, _, pi, _, _ = compute(payoff_tables, alpha=alpha, m=m)
      pi_list = np.append(pi_list, np.reshape(pi, (-1, 1)), axis=1)
      alpha_list.append(alpha)
      # Stop when pi converges
      if num_iters > 0 and np.allclose(pi, pi_list[:, num_iters - 1], rtol,
                                       atol):
        break
      alpha *= alpha_mult_factor
      num_iters += 1
      alpharank_succeeded_once = True
    except ValueError as _:
      if warm_start_alpha is not None and not alpharank_succeeded_once:
        # When warm_start_alpha is used, there's a chance that
        # the initial warm_start_alpha is too large and causes exceptions due to
        # the Markov transition matrix being reducible. So keep decreasing until
        # a single success occurs.
        alpha /= 2
      elif not np.allclose(pi_list[:, -1], pi_list[:, -2], rtol, atol):
        # Sweep stopped due to multiple stationary distributions, but pi had
        # not converged due to the alpha scaling being too large.
        alpha /= alpha_mult_factor
        alpha_mult_factor = (alpha_mult_factor + 1.) / 2.
        alpha *= alpha_mult_factor
      else:
        break

  if strat_labels is None:
    strat_labels = utils.get_strat_profile_labels(payoff_tables,
                                                  payoffs_are_hpt_format)
  fig = plot_pi_vs_alpha(
      pi_list.T,
      alpha_list,
      num_populations,
      num_strats_per_population,
      strat_labels,
      num_strats_to_label=10)

  return pi, alpha, fig


def compute_and_report_alpharank(payoff_tables,
                                 m=50,
                                 alpha=100,
                                 verbose=False,
                                 num_top_strats_to_print=8):
  """Computes and visualizes Alpha-Rank outputs.

  Args:
    payoff_tables: List of game payoff tables, one for each agent identity. Each
      payoff_table may be either a numpy array, or a _PayoffTableInterface
      object.
    m: Finite population size.
    alpha: Fermi distribution temperature parameter.
    verbose: Set to True to print intermediate results.
    num_top_strats_to_print: Number of top strategies to print.

  Returns:
    pi: AlphaRank stationary distribution/rankings.
  """
  payoffs_are_hpt_format = utils.check_payoffs_are_hpt(payoff_tables)
  rhos, rho_m, pi, _, _ = compute(payoff_tables, m=m, alpha=alpha)
  strat_labels = utils.get_strat_profile_labels(payoff_tables,
                                                payoffs_are_hpt_format)

  if verbose:
    print_results(payoff_tables, payoffs_are_hpt_format, pi=pi)

  utils.print_rankings_table(
      payoff_tables,
      pi,
      strat_labels,
      num_top_strats_to_print=num_top_strats_to_print)
  m_network_plotter = NetworkPlot(
      payoff_tables, rhos, rho_m, pi, strat_labels, num_top_profiles=8)
  m_network_plotter.compute_and_draw_network()
  return pi