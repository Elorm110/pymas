"""pymas: lightweight Python statistics library."""
import math
import random
import statistics as _stats
from collections import Counter

def _vals(data):
    v=list(data)
    if not v: raise ValueError("data must not be empty")
    try: return [float(x) for x in v]
    except (TypeError, ValueError): raise TypeError("data must contain numeric values")

def _pair(x,y):
    a,b=_vals(x),_vals(y)
    if len(a)!=len(b): raise ValueError("x and y must have the same length")
    return a,b

def mean(data):
    v=_vals(data); return sum(v)/len(v)

def median(data):
    v=sorted(_vals(data)); n=len(v); m=n//2
    return v[m] if n%2 else (v[m-1]+v[m])/2

def mode(data):
    v=list(data)
    if not v: raise ValueError("data must not be empty")
    c=Counter(v); high=max(c.values())
    if high==1: return None
    modes=[x for x,n in c.items() if n==high]
    return modes[0] if len(modes)==1 else modes

def geometric_mean(data):
    v=_vals(data)
    if any(x<=0 for x in v): raise ValueError("values must be positive")
    return math.exp(sum(math.log(x) for x in v)/len(v))

def harmonic_mean(data):
    v=_vals(data)
    if any(x==0 for x in v): raise ValueError("zero is not allowed")
    return len(v)/sum(1/x for x in v)

def trimmed_mean(data,percent):
    v=sorted(_vals(data))
    if not 0<=percent<50: raise ValueError("percent must be between 0 and 50")
    k=int(len(v)*percent/100); v=v[k:len(v)-k] if k else v
    return sum(v)/len(v)

def range(data):
    v=_vals(data); return max(v)-min(v)

def variance(data):
    v=_vals(data); m=mean(v); return sum((x-m)**2 for x in v)/len(v)

def sample_variance(data):
    v=_vals(data)
    if len(v)<2: raise ValueError("at least 2 values required")
    m=mean(v); return sum((x-m)**2 for x in v)/(len(v)-1)

def standard_deviation(data): return math.sqrt(variance(data))
def sample_standard_deviation(data): return math.sqrt(sample_variance(data))

def mean_absolute_deviation(data):
    v=_vals(data); m=mean(v); return sum(abs(x-m) for x in v)/len(v)

def minimum(data): return min(_vals(data))
def maximum(data): return max(_vals(data))

def percentile(data,p):
    v=sorted(_vals(data))
    if not 0<=p<=100: raise ValueError("p must be between 0 and 100")
    pos=(len(v)-1)*p/100; lo,hi=math.floor(pos),math.ceil(pos)
    return v[lo] if lo==hi else v[lo]+(v[hi]-v[lo])*(pos-lo)

def quartile(data,q):
    if not 0<=q<=4: raise ValueError("q must be between 0 and 4")
    return percentile(data,q*25)

def interquartile_range(data): return percentile(data,75)-percentile(data,25)

def coefficient_of_variation(data):
    m=mean(data)
    if m==0: raise ZeroDivisionError("mean cannot be zero")
    return standard_deviation(data)/m

def percentile_rank(data,value):
    v=_vals(data); return 100*(sum(x<value for x in v)+0.5*sum(x==value for x in v))/len(v)

def frequency(data): return dict(Counter(list(data)))

def relative_frequency(data):
    v=list(data)
    if not v: raise ValueError("data must not be empty")
    c=Counter(v); return {k:n/len(v) for k,n in c.items()}

def cumulative_frequency(data):
    c=Counter(sorted(list(data))); total=0; out={}
    for k in sorted(c):
        total+=c[k]; out[k]=total
    return out

def covariance(x,y):
    x,y=_pair(x,y); mx,my=mean(x),mean(y)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/len(x)

def correlation(x,y):
    x,y=_pair(x,y); sx,sy=standard_deviation(x),standard_deviation(y)
    if sx==0 or sy==0: raise ZeroDivisionError("constant data")
    return covariance(x,y)/(sx*sy)

def linear_regression(x,y):
    x,y=_pair(x,y); mx,my=mean(x),mean(y)
    den=sum((a-mx)**2 for a in x)
    if den==0: raise ZeroDivisionError("x cannot be constant")
    slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/den
    return {"slope":slope,"intercept":my-slope*mx}

def r_squared(x,y): return correlation(x,y)**2

def probability(successes,total):
    if total<=0 or successes<0 or successes>total: raise ValueError("invalid counts")
    return successes/total

def conditional_probability(a_and_b,b):
    if b<=0 or a_and_b<0 or a_and_b>b: raise ValueError("invalid probabilities")
    return a_and_b/b

def bayes(prior,likelihood,evidence):
    if not 0<=prior<=1 or not 0<=likelihood<=1 or not 0<evidence<=1: raise ValueError("invalid probability")
    return prior*likelihood/evidence

def expected_value(values,probabilities):
    v,p=_pair(values,probabilities)
    if any(x<0 for x in p) or not math.isclose(sum(p),1,abs_tol=1e-9): raise ValueError("probabilities must sum to 1")
    return sum(x*q for x,q in zip(v,p))

def normal_pdf(x,mean,std):
    if std<=0: raise ValueError("std must be positive")
    return math.exp(-0.5*((x-mean)/std)**2)/(std*math.sqrt(2*math.pi))

def normal_cdf(x,mean,std):
    if std<=0: raise ValueError("std must be positive")
    return 0.5*(1+math.erf((x-mean)/(std*math.sqrt(2))))

def binomial(n,p,k):
    if not isinstance(n,int) or n<0 or not 0<=p<=1: raise ValueError("invalid n or p")
    if not isinstance(k,int) or k<0 or k>n: return 0.0
    return math.comb(n,k)*p**k*(1-p)**(n-k)

def poisson(lam,k):
    if lam<0: raise ValueError("lambda must be non-negative")
    if not isinstance(k,int) or k<0: return 0.0
    return math.exp(-lam)*lam**k/math.factorial(k)

def uniform(a,b,x):
    if b<=a: raise ValueError("b must be greater than a")
    return 1/(b-a) if a<=x<=b else 0.0

def z_score(value,mean,std):
    if std<=0: raise ValueError("std must be positive")
    return (value-mean)/std

def z_test(sample_mean,population_mean,std,n):
    if std<=0 or n<=0: raise ValueError("invalid std or n")
    z=(sample_mean-population_mean)/(std/math.sqrt(n))
    return {"z":z,"p_value":2*(1-normal_cdf(abs(z),0,1))}

def t_test(sample1,sample2):
    a,b=_vals(sample1),_vals(sample2)
    if len(a)<2 or len(b)<2: raise ValueError("each sample needs 2+ values")
    ma,mb=mean(a),mean(b); va,vb=sample_variance(a),sample_variance(b)
    se=math.sqrt(va/len(a)+vb/len(b))
    if se==0: raise ZeroDivisionError("standard error is zero")
    t=(ma-mb)/se
    df=(va/len(a)+vb/len(b))**2/((va/len(a))**2/(len(a)-1)+(vb/len(b))**2/(len(b)-1))
    # Two-sided normal approximation for a simple dependency-free result.
    p=2*(1-normal_cdf(abs(t),0,1))
    return {"t":t,"df":df,"p_value":p}

def chi_square(observed,expected):
    o,e=_pair(observed,expected)
    if any(x<=0 for x in e): raise ValueError("expected values must be positive")
    return {"chi_square":sum((a-b)**2/b for a,b in zip(o,e)),"df":len(o)-1}

def anova(groups):
    g=[_vals(x) for x in groups]
    if len(g)<2 or any(len(x)<2 for x in g): raise ValueError("need 2+ groups with 2+ values")
    allv=[x for group in g for x in group]; grand=mean(allv)
    between=sum(len(x)*(mean(x)-grand)**2 for x in g)
    within=sum(sum((x-mean(group))**2 for x in group) for group in g)
    dfb=len(g)-1; dfw=len(allv)-len(g); msw=within/dfw
    return {"f":between/dfb/msw if msw else math.inf,"df_between":dfb,"df_within":dfw}

def margin_of_error(std,n,confidence=0.95):
    if std<0 or n<=0 or not 0<confidence<1: raise ValueError("invalid arguments")
    z=_stats.NormalDist().inv_cdf((1+confidence)/2)
    return z*std/math.sqrt(n)

def confidence_interval_mean(data,confidence=0.95):
    m=mean(data); moe=margin_of_error(standard_deviation(data),len(_vals(data)),confidence)
    return (m-moe,m+moe)

def random_sample(data,size):
    v=list(data)
    if not 0<=size<=len(v): raise ValueError("invalid size")
    return random.sample(v,size)

def bootstrap(data,size):
    v=list(data)
    if not v or size<0: raise ValueError("invalid data or size")
    return [random.choice(v) for _ in range(size)]

def shuffle(data):
    v=list(data); random.shuffle(v); return v

def sort(data): return sorted(list(data))

def remove_outliers(data):
    v=_vals(data); q1,q3=percentile(v,25),percentile(v,75); iqr=q3-q1
    return [x for x in v if q1-1.5*iqr<=x<=q3+1.5*iqr]

def normalize(data):
    v=_vals(data); lo,hi=min(v),max(v)
    return [0.0]*len(v) if hi==lo else [(x-lo)/(hi-lo) for x in v]

def standardize(data):
    v=_vals(data); sd=standard_deviation(v); m=mean(v)
    return [0.0]*len(v) if sd==0 else [(x-m)/sd for x in v]
