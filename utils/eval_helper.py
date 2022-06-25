###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import copy
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures
from random import shuffle
from datetime import datetime
from scipy.linalg import eigvalsh
from utils.dist_helper import compute_mmd, gaussian_emd, gaussian, emd, gaussian_tv
from sklearn import metrics

PRINT_TIME = False
__all__ = [
     'degree_stats', 'clustering_stats', 'spectral_stats',
]


def degree_worker(G):
  return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
  ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for deg_hist in executor.map(degree_worker, graph_ref_list):
        sample_ref.append(deg_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
        sample_pred.append(deg_hist)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for deg_hist in executor.map(degree_worker, graph_ref_list):
    #     sample_ref.append(deg_hist)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
    #     sample_pred.append(deg_hist)
  else:
    for i in range(len(graph_ref_list)):
      degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
      sample_ref.append(degree_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      degree_temp = np.array(
          nx.degree_histogram(graph_pred_list_remove_empty[i]))
      sample_pred.append(degree_temp)
  # print(len(sample_ref), len(sample_pred))

  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
  mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist

###############################################################################

def spectral_worker(G):
  # eigs = nx.laplacian_spectrum(G)
  eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
  spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
  spectral_pmf = spectral_pmf / spectral_pmf.sum()
  # from scipy import stats
  # kernel = stats.gaussian_kde(eigs)
  # positions = np.arange(0.0, 2.0, 0.1)
  # spectral_density = kernel(positions)

  # import pdb; pdb.set_trace()
  return spectral_pmf

def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
  ''' Compute the distance between the eigenvalue distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
  sample_ref = []
  sample_pred = []
  # in case an empty graph is generated
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_ref_list):
        sample_ref.append(spectral_density)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        sample_pred.append(spectral_density)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
    #     sample_ref.append(spectral_density)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
    #     sample_pred.append(spectral_density)
  else:
    for i in range(len(graph_ref_list)):
      spectral_temp = spectral_worker(graph_ref_list[i])
      sample_ref.append(spectral_temp)
    for i in range(len(graph_pred_list_remove_empty)):
      spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
      sample_pred.append(spectral_temp)
  # print(len(sample_ref), len(sample_pred))

  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
  mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
  # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing degree mmd: ', elapsed)
  return mmd_dist

###############################################################################

def clustering_worker(param):
  G, bins = param
  clustering_coeffs_list = list(nx.clustering(G).values())
  hist, _ = np.histogram(
      clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
  return hist


def clustering_stats(graph_ref_list,
                     graph_pred_list,
                     bins=100,
                     is_parallel=True):
  sample_ref = []
  sample_pred = []
  graph_pred_list_remove_empty = [
      G for G in graph_pred_list if not G.number_of_nodes() == 0
  ]

  prev = datetime.now()
  if is_parallel:
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for clustering_hist in executor.map(clustering_worker,
                                          [(G, bins) for G in graph_ref_list]):
        sample_ref.append(clustering_hist)
    with concurrent.futures.ThreadPoolExecutor() as executor:
      for clustering_hist in executor.map(
          clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
        sample_pred.append(clustering_hist)

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for clustering_hist in executor.map(clustering_worker,
    #                                       [(G, bins) for G in graph_ref_list]):
    #     sample_ref.append(clustering_hist)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #   for clustering_hist in executor.map(
    #       clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
    #     sample_pred.append(clustering_hist)

    # check non-zero elements in hist
    #total = 0
    #for i in range(len(sample_pred)):
    #    nz = np.nonzero(sample_pred[i])[0].shape[0]
    #    total += nz
    #print(total)
  else:
    for i in range(len(graph_ref_list)):
      clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
      hist, _ = np.histogram(
          clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
      sample_ref.append(hist)

    for i in range(len(graph_pred_list_remove_empty)):
      clustering_coeffs_list = list(
          nx.clustering(graph_pred_list_remove_empty[i]).values())
      hist, _ = np.histogram(
          clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
      sample_pred.append(hist)

  # mmd_dist = compute_mmd(
  #     sample_ref,
  #     sample_pred,
  #     kernel=gaussian_emd,
  #     sigma=1.0 / 10,
  #     distance_scaling=bins)

  mmd_dist = compute_mmd(
      sample_ref,
      sample_pred,
      kernel=gaussian_tv,
      sigma=1.0 / 10)

  elapsed = datetime.now() - prev
  if PRINT_TIME:
    print('Time computing clustering mmd: ', elapsed)
  return mmd_dist







