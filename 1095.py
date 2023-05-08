
import numpy as np

import itertools

import scipy
import scipy.stats

#approximate F_{\inf}


def pmean(x, p):
  if p <= 0 and np.min(x) == 0:
    return np.min(x)
  elif p == -np.inf:
    return np.min(x)
  elif p == 0:
    return np.exp(np.mean(np.log(x)))
  elif p == np.inf:
    return np.max(x)
  else:
    return np.mean(x ** p) ** (1 / p)
  
  #x[i,1] = np.mean(x[i-1])
  #x[i,2] = np.sqrt(np.mean(x[i-1]**2))


def wpmean(x, p, w):
  if p <= 0 and np.min(x) == 0:
    return np.min(x)
  elif p == -np.inf:
    return np.min(x)
  elif p == 0:
    return np.exp(np.sum(w * np.log(x)))
  elif p == np.inf:
    return np.max(x)
  elif np.abs(p) < 0.1:
    #Numerically stable way to handle pth powers and roots for p \approx 0
    return np.exp( np.log1p( np.sum( w * np.expm1( p * np.log(x) ) ) ) / p )
  elif np.abs(p) > 10:
  #elif p > 10:
    #Somewhat numerically stable way to handle pth powers and roots for p >> 0
    return np.sum( np.exp( p * np.log(x) + np.log(w) ) ) ** (1 / p)
  else:
    #return np.sum( np.exp( p * np.log(x) + np.log(w) ) ) ** (1 / p)
    return np.sum(w * x ** p) ** (1 / p)

def wpmean_stable(x, p, w):
  #Using multiplicative linearity, compute a weighted power-mean of a normalized sentiment vector.
  scale = 1
  if p > 10:
    #Max normalization
    scale = np.max(w)
  elif p < -10:
    #Max normalization
    scale = np.min(w)
  #if np.abs(p) < 10:
  elif np.abs(p) > 1:
    #Utilitarian normalization
    scale = np.dot(w, x)

  return scale * wpmean(x / scale, p, w)


##############################
#Meta-fairness and convergence

def run_exper(x0, pm, iterations):
  ng = x0.shape[0]
  x = np.zeros((iterations, ng), np.float64)
  x[0] = x0
  for i in range(1, iterations):
    #print([i] + list(x))
    for j in range(ng):
      x[i,j] = pmean(x[i-1], pm[j])

    """
    x[i,0] = np.min(x[i-1])
    x[i,1] = np.mean(x[i-1])
    x[i,2] = np.sqrt(np.mean(x[i-1]**2))
    x[i,3] = np.max(x[i-1]**2)
    """
  #print(x0)
  #desc = "$p^{(\\Meta)} = " + "\\{" + f"{str(pm)[1:-1]}" + "\\}$"
  desc = "$p^{(\\Meta)} = " + "\\langle " + f"{str(pm)[1:-1]}" + " \\rangle$"

  return (desc, x)

def write_exper(iterations, x, lf):
  ng = x.shape[1]

  line = f"iteration"
  for j in range(ng):
    line += f",g{j}"
  line += "\n"
  lf.write(line)

  for i in range(iterations):
    line = f"{i}"
    for j in range(ng):
      line += f",{x[i][j]}"
    line += "\n"
    lf.write(line)



#Old meta-fairness convergence experiments
def run_meta_expers():
  iterations = 16
  #x0 = np.asarray([1, 2, 2, 4])
  #pm = [-1, 0, 1, 2]

  #x0 = np.asarray([1, 2, 4, 8, 16])
  x0 = np.asarray([0.25, 0.5, 1, 2, 4])
  pm = [-2, -1, 0, 1, 2]


  with open("plots/a.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  x0 = np.asarray([1, 2, 3, 4, 5])
  pm = [-2, -1, 0, 1, 2]

  with open("plots/b.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  iterations = 256 #128 #64

  x0 = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8])
  pm = [0, 1, 2, 3, 4, 5, 6, 7, 8]

  with open("plots/c.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  #iterations = 128 #64

  x0 = np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
  pm = [-1, 0, 1, 2, 3, 4, 5, 6, 7, np.inf]

  with open("plots/d.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  iterations = 32
  x0 = np.asarray([1, 2, 3, 4, 5])
  pm = [-np.inf, 0, 1, 2, np.inf]

  with open("plots/e.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  print("Running E Random")


  for rand_iter in range(10):
    x0[1:-1] = np.random.uniform(low=np.min(x0),high=np.max(x0),size=x0.shape[0]-2)

    with open(f"plots/e_rand_{rand_iter}.csv", "w") as lf:
      (desc, x) = run_exper(x0, pm, iterations)
      print(desc)
      write_exper(iterations, x, lf)

  print("Running ELH")

  x0[1:-1] = x0[0]

  with open(f"plots/e_lo.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)

  x0[1:-1] = x0[-1]

  with open(f"plots/e_hi.csv", "w") as lf:
    (desc, x) = run_exper(x0, pm, iterations)
    print(desc)
    write_exper(iterations, x, lf)
      


def run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_offsets=None, dist_scales=None, dist_type="uniform"):

  #dist_type="beta"
  #dist_type="bernoulli"

  g = len(ws[0]) #Number of groups
  
  #pmean_params = itertools.product(ps, ws)
  #for pi in ps:
  #  for wi in ws:
  
  #quantiles = np.linspace(0, 1, num=7, endpoint=False)
  #quantiles = np.linspace(0.125, 0.875, num=7, endpoint=True)le
  #quantiles = np.linspace(0, 1, num=9, endpoint=True)

  #quantiles = np.linspace(0, 1, num=11, endpoint=True)
  #quantiles = quantiles.tolist()

  quantiles = []
  
  #sigmarules = [0.6827, 0.9545, 0.9973]
  sigmarules = [0.6827]

  for si in sigmarules:
    quantiles.append((1 - si) / 2)
    quantiles.append(1 - (1 - si) / 2)

  #quantiles = np.asarray(sorted(quantiles))
  quantiles = np.asarray(quantiles)

  header = "m,b,p,w"
  header += ",b2,w2"
  #header += ",min,mean-stdev,mean,mean+stdev,max"
  header += ",mean-stdev,mean,mean+stdev"
  for qi in quantiles:
    header += f",q{qi:.5f}"
  header += ",stdev,g2bound,holderbound,sharpe,max/min,(max-min)/true,true"
  header += "\n"
  lf.write(header)
  
  lf.write(f"#w vectors: {ws.tolist()}\n")
  lf.write(f"#b vectors: {bs.tolist()}\n")

  for mi in ms:
    for (bii, bi) in enumerate(bs):
    
      if dist_type == "beta":
        beta_dists = [scipy.stats.beta(bij, 1 - bij) for bij in bi]
        dists = beta_dists
      elif dist_type == "bernoulli":
        bernoulli_dists = [scipy.stats.bernoulli(bij) for bij in bi]
        dists = bernoulli_dists
      elif dist_type == "uniform":
        uniform_dists = [scipy.stats.uniform(loc = bij - 0.5, scale=1.0) for bij in bi]
        dists = uniform_dists


      if dist_scales != None:
        this_scales = dist_scales[bii]
      else:
        this_scales = np.ones(g)

      if dist_offsets != None:
        this_offsets = this_offsets[bii]
      else:
        this_offsets = np.zeros(g)

      """
      if dist_scales != None:
        dists = [{**dists[di], "scale" : di.scale * this_scales[dii]} for dii in range(len(dists))]
      """

      means = np.asarray([di.expect() for di in dists]) * this_scales + this_offsets
      variances = np.asarray([di.var() for di in dists]) * np.square(this_scales)

      this_emeans = np.zeros((n_trials, g), np.float128)
      for ni in range(n_trials):
        for gi in range(g):
          #this_emeans[ni,gi] = np.random.binomial(mi, bi[gi], size=1) / mi
          #this_emeans[ni,gi] = np.mean(np.random.uniform(0, 2 * bi[gi], size=mi))
          if bi[gi] > 0:
            pass
            #this_emeans[ni,gi] = np.mean(np.random.beta(bi[gi], 1 - bi[gi], size=mi))
            #this_emeans[ni,gi] = np.mean(np.random.beta(bi[gi] / 2, (1 - bi[gi]) / 2, size=mi))

          this_emeans[ni,gi] = np.mean(dists[gi].rvs(mi))

      #print(bi, means, this_emeans)

      this_emeans = this_emeans * this_scales + this_offsets
      this_emeans = np.maximum(this_emeans, 0)

      #for (pi, wi) in pmean_params:
      for (pii, pi) in enumerate(ps):
        for (wii, wi) in enumerate(ws):
          this_pmeans = np.zeros(n_trials, np.float128)
          for ni in range(n_trials):
            this_pmeans[ni] = wpmean(this_emeans[ni], pi, wi)
          true_mean = wpmean(bi, pi, wi)
          avg = np.mean(this_pmeans)
          stdev = np.std(this_pmeans, ddof=1)
          qs = list(np.quantile(this_pmeans, quantiles)) #method='linear'
          #this_stats = [np.min(this_pmeans), avg - stdev, avg, avg + stdev, np.max(this_pmeans)]

          #Holder continuity bounds:
          #v_scale = 1 #For Bernoulli distributions
          #v_scale = 0.5 #For Beta distribution s.t. a + b = 1
          #epsilon = np.sqrt( v_scale * bi[1] * (1 - bi[1]) / mi )
          epsilon = np.sqrt( variances[1] / mi )

          wmin = wi[1] #Use group 2 weight.
          #wmin = np.min(wi)

          lam = 1
          alpha = wmin
          err = lam * epsilon ** alpha

          if pi > 0:
            #lam = 1 / pi
            lam = wmin / pi #Using H\"older bound specific to minority group
            alpha = pi
            err2 = lam * epsilon ** alpha
            err = min(err, err2)

          elif pi < 0:
            lam = wmin ** (-1 / np.abs(pi))
            alpha = 1
            err2 = lam * epsilon ** alpha
            err = min(err, err2)

          this_stats = []
          this_stats += [avg - stdev, avg, avg + stdev]
          this_stats += qs
          this_stats += [stdev, epsilon, err, avg/stdev, qs[-1] / qs[0], (qs[-1] - qs[0]) / true_mean, true_mean]

          line = f"{mi},{bii},{pi},{wii}"
          line += f",{bi[1]},{wi[1]}"
          for si in this_stats:
            line += f",{si}"
          line += "\n"
          lf.write(line)


def get_default_params(dist_type="uniform"):
  if dist_type == "uniform":
    ps = np.asarray([0.0])
    ws = np.asarray([[2/3, 1/3]])
    ms = np.asarray([100])
    
    bs = np.asarray([[0.999, 0.001]])
    #bs = np.asarray([[0.995, 0.005]])

  else:
    ps = np.asarray([0.0])
    #ws = np.asarray([[2/3, 1/3]])
    ws = np.asarray([[0.5, 0.5]])
    #ws = np.asarray([[0.5, 0.5], [2/3, 1/3], [0.75, 0.25]])
    #ms = np.asarray([250])
    #ms = np.asarray([200])
    #ms = np.asarray([150])
    #ms = np.asarray([100])
    ms = np.asarray([50])
    #bs = np.asarray([[0.5, 0.01]])
    #bs = np.asarray([[0.853553, 0.01]]) #Variance 1/8
    #bs = np.asarray([[0.933013, 0.01]]) #Variance 1/16
    #bs = np.asarray([[1, 0.025]])
    #bs = np.asarray([[1, 0.02]])
    #bs = np.asarray([[1, 0.015]])
    #bs = np.asarray([[1, 0.01]])
    
    #bs = np.asarray([[0.985, 0.015]]) #Variance ?
    bs = np.asarray([[0.99, 0.01]]) #Variance ?
    #bs = np.asarray([[0.999, 0.001]]) #Variance ?

  return (ps, ws, ms, bs)

def run_bernoulli_experiments():

  dist_type = "bernoulli"
  directory = f"./exper/{dist_type}/"

  #n_trials = 512 #256
  #n_trials = 1024
  #n_trials = 2048
  #n_trials = 4096
  n_trials = 5000
  
  #grid_factor = 500
  grid_factor = 100


  #Testing:
  #n_trials = 100
  #grid_factor = 10


  with open(directory + "changep.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ps = np.linspace(0, 0.5, num=5, endpoint=True)
    #ps = np.linspace(0, 1, num=101, endpoint=True)
    #ps = np.linspace(-1, 1, num=201, endpoint=True)
    ps = np.tan(np.linspace(-np.pi/2*0.999999, 0, num=grid_factor+1, endpoint=True)) + 1

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changew.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 0.5, num=grid_factor // 2 + 1, endpoint=True)])
    ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 1, num=grid_factor + 1, endpoint=True)])
    #ms = np.asarray([200])

    ms = np.asarray([100])
    bs = np.asarray([[0.999, 0.001]]) #Variance ?

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changeb.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    bmax = bs[0,0]
    bs = np.asarray([[bmax, bi] for bi in np.linspace(0 + 1e-10, bmax - 1e-10, num=int(round(bmax*grid_factor))+1, endpoint=True)])
    ms = np.asarray([20])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  """
  with open(directory + "changeb1.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    ps[0] = 1.0
    bmax = bs[0,0]
    bs = np.asarray([[bmax, wi] for wi in np.linspace(0, bmax, num=int(round(bmax*grid_factor))+1, endpoint=True)])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)
  """

  with open(directory + "changem.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ms = [2 ** i for i in range(8, 16+1)]
    #ms = [2 ** i for i in range(6, 16+1)]
    #ms = [2 ** i for i in range(1, 16+1)]
    m_split = 3
    ms = [int(round(10 ** (i / m_split))) for i in range(0*1*m_split, 6*m_split+1)]

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)



def run_beta_experiments():

  dist_type = "beta"
  directory = f"./exper/{dist_type}/"

  #n_trials = 512 #256
  #n_trials = 1024
  #n_trials = 2048
  #n_trials = 4096
  n_trials = 5000
  
  #grid_factor = 500
  grid_factor = 100


  #Testing:
  #n_trials = 100
  #grid_factor = 10


  with open(directory + "changep.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ps = np.linspace(0, 0.5, num=5, endpoint=True)
    #ps = np.linspace(0, 1, num=101, endpoint=True)
    #ps = np.linspace(-1, 1, num=201, endpoint=True)
    ps = np.tan(np.linspace(-np.pi/2*0.999999, 0, num=grid_factor+1, endpoint=True)) + 1

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changew.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 0.5, num=grid_factor // 2 + 1, endpoint=True)])
    ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 1, num=grid_factor + 1, endpoint=True)])
    #ms = np.asarray([200])

    ms = np.asarray([100])
    bs = np.asarray([[0.999, 0.001]]) #Variance ?

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changeb.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    bmax = bs[0,0]
    bs = np.asarray([[bmax, bi] for bi in np.linspace(0 + 1e-10, bmax - 1e-10, num=int(round(bmax*grid_factor))+1, endpoint=True)])
    ms = np.asarray([20])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  """
  with open(directory + "changeb1.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    ps[0] = 1.0
    bmax = bs[0,0]
    bs = np.asarray([[bmax, wi] for wi in np.linspace(0, bmax, num=int(round(bmax*grid_factor))+1, endpoint=True)])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)
  """

  with open(directory + "changem.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ms = [2 ** i for i in range(8, 16+1)]
    #ms = [2 ** i for i in range(6, 16+1)]
    #ms = [2 ** i for i in range(1, 16+1)]
    m_split = 3
    ms = [int(round(10 ** (i / m_split))) for i in range(0*1*m_split, 6*m_split+1)]

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)


def run_uniform_experiments():

  dist_type = "uniform"
  directory = f"./exper/{dist_type}/"

  #n_trials = 512 #256
  #n_trials = 1024
  #n_trials = 2048
  n_trials = 5000
  #grid_factor = 500
  grid_factor = 100


  #Testing:
  #n_trials = 500
  #grid_factor = 10
  #grid_factor = 50


  with open(directory + "changep.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ps = np.linspace(0, 0.5, num=5, endpoint=True)
    #ps = np.linspace(0, 1, num=101, endpoint=True)
    #ps = np.linspace(-1, 1, num=201, endpoint=True)
    ps = np.tan(np.linspace(-np.pi/2*0.999999, 0, num=grid_factor+1, endpoint=True)) + 1

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changew.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 0.5, num=grid_factor // 2 + 1, endpoint=True)])
    ws = np.asarray([[1 - wi, wi] for wi in np.linspace(0, 1, num=grid_factor + 1, endpoint=True)])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  with open(directory + "changeb.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    bmax = bs[0,0]
    bs = np.asarray([[bmax, bi] for bi in np.linspace(0, bmax, num=int(round(bmax*grid_factor))+1, endpoint=True)])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

  """
  with open(directory + "changeb1.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    ps[0] = 1.0
    bmax = bs[0,0]
    bs = np.asarray([[bmax, wi] for wi in np.linspace(0, bmax, num=int(round(bmax*grid_factor))+1, endpoint=True)])

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)
  """

  with open(directory + "changem.csv", "w") as lf:
    (ps, ws, ms, bs) = get_default_params(dist_type=dist_type)
    #ms = [2 ** i for i in range(8, 16+1)]
    #ms = [2 ** i for i in range(6, 16+1)]
    #ms = [2 ** i for i in range(1, 16+1)]
    m_split = 2
    ms = [int(round(10 ** (i / m_split))) for i in range(0*1*m_split, 6*m_split+1)]

    run_bandit_experiment(ps, ws, ms, bs, n_trials, lf, dist_type=dist_type)

print("Sorry for the mess, good luck with this code!")

run_bernoulli_experiments()
run_beta_experiments()
run_uniform_experiments()


