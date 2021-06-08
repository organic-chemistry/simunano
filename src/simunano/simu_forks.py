
import numpy as np
from collections import namedtuple
from pomegranate import GeneralMixtureModel,NormalDistribution
import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())


origin = namedtuple("origin",["pos","firing_time","L_fork_speed","R_fork_speed"])

def track(time,start_time=2,end_time=15,maxv=0.8,minv=0.1,inct=1,pulselen=4,dect=5):
    """
    Given a 1D array of time , generate a single fork
    following an exponential increasing law between start_time and start_time + pulselen
    followed by a decreasing exponential law.
    The ascending part is governed by maxv and inct (the characteristic time of the exponential)
    The descending part by minv and dect (the characteristic time of the exponential)
    It return a 1d array with the results, as well a the actual length of the ascending
    part (that can be truncated) and the length of the background part before the ascending
    exponential
    """
    before = time[time<=start_time]
    initial = time[ (time > start_time) & (time < start_time + pulselen)]
    if len(initial) != 0:
        initial = maxv*(1-np.exp(-(initial-start_time)/inct))
    final = time[(time >= start_time + pulselen) & (time < end_time)]
    if len(initial) != 0:
        startv = initial[-1]
    else:
        startv = maxv*(1-np.exp(-(pulselen)/inct))

    if len(final) != 0:
        final = startv + (minv -startv)*(1-np.exp(-(final-start_time-pulselen)/dect))
    end = time[time >= end_time]

    result = np.concatenate([np.zeros_like(before),initial,final,np.zeros_like(end)])
    #print(maxv,np.max(result))

    return result,len(initial),len(before)

def intersection(p1,p2):
    """
    Given two converging forks and their firing time and speeds,
    compute the position of the intersection
    as well as the position of the time of intersection.
    If the intersection is outside [x1,x2], the initial position of the forks,
    then return False
    """
    x1,t1,R_fork_speed=p1.pos,p1.firing_time,p1.R_fork_speed
    x2,t2,L_fork_speed=p2.pos,p2.firing_time,p2.L_fork_speed

    assert(x2>x1)

    #x = (x1+x2)/2 + (t2-t1)*v/2
    x = 1/(1/L_fork_speed+1/R_fork_speed)*(t2-t1 + x1/L_fork_speed+x2/R_fork_speed)
    if  not( x1<x<x2):
        return False,[None,None]

    t = (x2-x1)/(R_fork_speed+L_fork_speed) + (t1 * R_fork_speed + t2 * L_fork_speed)/(R_fork_speed+L_fork_speed)

    return True,[x,t]


def generate_mrt(pos_time,end=1000,start_at_zero=True):
    """
    Given a list of origin and firing times and fork speed
    return a 1d arry with the times at which the replication occurs
    By default the lowest time is zero.
    To do so it build a list with position and time of initiation and termination
    and then use numpy linera interpolation function
    """
    #print(pos_time)

    x1,t1,L_fork_speed = pos_time[0].pos,pos_time[0].firing_time,pos_time[0].L_fork_speed
    first = [0,t1+x1/L_fork_speed]
    pos_with_terms = [first]
    for p1,p2 in zip(pos_time[:-1],pos_time[1:]):
        possible,inte = intersection(p1,p2)
        pos_with_terms.extend([[p1.pos,p1.firing_time],inte+[]])
        if not possible:
            return False

    if len(pos_time) == 1:
        p2 = pos_time[0]
    pos_with_terms.append([p2.pos,p2.firing_time])
    x2,t2,R_fork_speed=p2.pos,p2.firing_time,p2.R_fork_speed
    pos_with_terms.append([end,t2+(end-x2)/R_fork_speed])

    p = np.array(pos_with_terms)
    #print(p)
    mrt = np.interp(np.arange(end),p[:,0],p[:,1])
    if start_at_zero:
        return mrt-np.min(mrt)
    else:
        return mrt

def generate_rfd(pos_time,end=1000):
    """
    Given a list of origin and firing times and fork speed
    return the direction of replication
    """
    #print(pos_time)
    rfd = np.zeros(end)
    x1,t1,L_fork_speed = pos_time[0].pos,pos_time[0].firing_time,pos_time[0].L_fork_speed
    rfd[:x1] = -1

    for p1,p2 in zip(pos_time[:-1],pos_time[1:]):
        possible,inte = intersection(p1,p2)
        middle = int(round(inte[0],0))
        rfd[p1.pos:middle]=1
        rfd[middle:p2.pos]=-1
    if len(pos_time) == 1:
        x2,t2=x1,t1
    else:
        x2,t2=p2.pos,p2.firing_time
    rfd[x2:]=1
    return rfd

def generate_track(pos_time,start_time=10,end=1000,params={},same_parameters=True):
    """
    Given a list of origin and firing times and fork speed
    and a start_time for the injection of Brdu return the incorporation
    of Brdu corresponding.
    """

    param_k = ["maxv","minv","pulselen","inct","dect"]
    list_param_generated=[]

    def generate_params(param_k,already_done={}):
        if already_done != {}:
            return already_done
        kw={}
        for p in param_k:
            if type(params[p]) == list:
                kw[p] = params[p][0] + (params[p][1]-params[p][0])*np.random.rand()
            else:
                kw[p] = params[p]
        list_param_generated.append(kw)
        return kw

    kw = {}
    if same_parameters:
        kw = generate_params(param_k)



    trac = np.zeros(end)
    x1,t1,L_fork_speed = pos_time[0].pos,pos_time[0].firing_time,pos_time[0].L_fork_speed
    time= np.arange(t1,t1+x1/L_fork_speed,1/L_fork_speed)

    t,len_init,before = track(time,start_time=start_time,end_time=t1+x1/L_fork_speed,
                               **generate_params(param_k,kw))
    trac[:x1] = t[:x1][::-1]

    #print(len_init)
    len_initial = [len_init + 0] #store the length of the increasing parts
    pos_s = [[x1-len_init-before,x1-before]]
    for p1,p2 in zip(pos_time[:-1],pos_time[1:]):
        possible,inte = intersection(p1,p2)
        middle = int(round(inte[0]))


        size = len(trac[p1[0]:middle])
        starto = p1.firing_time
        R_fork_speed = p1.R_fork_speed
        time= np.arange(starto,starto+size/R_fork_speed,1/R_fork_speed)
        #print(time)
        #print(time,len(time))
        #print(p1[0],p2[0],middle)
        #print(track(time,start_time=start_time,end_time=starto+size/v)[:size])
        trac[p1.pos:middle]
        t,len_init,before= track(time,start_time=start_time,end_time=starto+size/R_fork_speed,
                                   **generate_params(param_k,kw))
        trac[p1.pos:middle] = t[:size]
        len_initial.append(len_init + 0)
        pos_s += [[p1.pos+before,p1.pos+len_init+before]]

        size = len(trac[middle:p2.pos])
        starto = p2.firing_time
        L_fork_speed = p2.L_fork_speed
        time= np.arange(starto,starto+size/L_fork_speed,1/L_fork_speed)
        #print(time,len(time))
        trac[middle:p2.pos]

        t,len_init,before = track(time,start_time=start_time,end_time=starto+size/L_fork_speed,
                                   **generate_params(param_k,kw))
        trac[middle:p2.pos] = t[:size][::-1]
        len_initial.append(len_init + 0)
        pos_s += [[p2.pos-len_init-before,p2.pos-before]]


    if len(pos_time) == 1:
        x2,t2 = x1,t1
        R_fork_speed =  pos_time[0].R_fork_speed
    else:
        x2,t2=p2.pos,p2.firing_time
        R_fork_speed = p2.R_fork_speed
    size = len(trac[x2:])
    time= np.arange(t2,t2+size/R_fork_speed,1/R_fork_speed)


    t,len_init,before = track(time,start_time=start_time,end_time=t2+size/R_fork_speed,
                               **generate_params(param_k,kw))
    trac[x2:] = t[:size]
    len_initial.append(len_init + 0)
    pos_s += [[x2+before,x2+len_init+before]]

    if not same_parameters:
        kw = list_param_generated
    return trac,[len_initial,pos_s],kw


def create_possible_origins(n_ori,n_sim,average_fork_speed,chlen,scaling=15):
    """
    generate a list of possible simulation given a number of origin an average_fork_speed
    """
    sims =[]
    while len(sims) != n_sim:

        pos = np.random.randint(0,chlen,n_ori)
        times = np.random.randint(0,chlen/n_ori/scaling,n_ori)
        times -= min(times)
        pos.sort()
        if len(set(pos)) != len(pos):
            continue
        #print(pos)
        sim = [origin(p,t,average_fork_speed,average_fork_speed) for p,t in zip(pos,times)]
        if type(generate_mrt(sim)) != bool:
            sims.append(sim)
    return sims

if __name__ == "__main__":

    import argparse
    import uuid
    import json
    from scipy import stats
    import pandas as pd
    import pylab
    np.random.seed(0)

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str,default="mock")
    parser.add_argument('--parameter_file', type=str,default='data/params.json')

    parser.add_argument('--average_distance_between_ori', type=float, default=50000)
    parser.add_argument('--multi',dest="one_fork", action="store_false")
    parser.add_argument('--correct_for_height', action="store_true")
    parser.add_argument('--ground_truth',  action="store_true")
    parser.add_argument('--fork_position',  action="store_true",
                        help="record fork positions")

    parser.add_argument('--resolution', type=int, default=100,
                        help="resolution in bp of the simulation")
    parser.add_argument('--n_conf_ori', type=int, default=400,
                        help="Generate set of ori and firing times")
    parser.add_argument('--time_per_mrt', type=int, default=400,
                            help="Generate time of starting pulse per configuration")
    parser.add_argument('--read_per_time', type=int, default=1,
                                help="Correspond to truncated fiber when the option whole_length"
                                     "is set to False")

    parser.add_argument('--draw_sample',  type=int,default=0)

    args = parser.parse_args()


    ##############################################
    # Generate track parameters
    with open(args.parameter_file,"r") as f:
        params = json.load(f)
    # maxv: lowest highest value of the plateau when increasing

    # minv:[0.12-0.05,0.15-0.05], #l owest highest value of the plateau when decreasing

    # pulselen":[2,2], # smallest longest value of the pulse length in minute
    # inct : [.25,1.25], # lowest highest value ofcharacteristic time of the increasing exponential
    # dect : [2.,5]

    #############################################
    #Either create ori at specific position and firing time
    average_fork_speed=15 # in 100 bp/min
    Sim = [[origin(50,2,average_fork_speed,average_fork_speed),
            origin(70,2,average_fork_speed,average_fork_speed)]]

    ##############################################
    #Choose fiber size and distributions
    resolution = args.resolution

    if not args.one_fork:
        chlen=300000 // resolution
        whole_length=False

    else:
        chlen = 50000 // resolution
        whole_length=True

    possiblesize = np.arange(5000//resolution,chlen)
    distribsize = stats.lognorm(0.5,scale=35000/resolution).pdf(possiblesize)
    distribsize /= np.sum(distribsize)

    nfork = {}
    pos={}
    fiber = {}
    rfd = {}
    gt = {}
    parameters = {}
    all_speeds={}
    positions = {}
    def draw(law):
        if law["type"] == "pomegranate":
            return GeneralMixtureModel.from_json(law["params"]).sample(1)
        if law["type"] == "choices":
            return np.random.choices(law["params"])
        if law["type"] == "uniform":
            return law["params"][0] + (law["params"][1]-law["params"][0])*np.random.rand()
        if law["type"] == "normal":
            return np.random.normal(loc=law["params"][0] ,scale=law["params"][1])
        if law["type"] == "exp":
            if "data" not in law:
                law["data"] = pd.read_csv(law["params"])["data"]
            which = int(np.random.randint(len(law["data"])))
            shift=0
            if "shift" in law:
                shift= law["shift"]
            return law["data"][which]+shift

    for sim_number  in range(args.n_conf_ori): # [current]:

        average_fork_speed = draw(params["speed"]) / resolution

        if average_fork_speed<=0:
            continue

        if not args.one_fork:
            sim = create_possible_origins(int(chlen / (args.average_distance_between_ori / resolution)),1,
                                  average_fork_speed,chlen)[0]
            mrt = generate_mrt(sim,end=chlen)
        # Draw time between the first 3/5 of the MRT
            minn = min(mrt)
            maxi = minn +  3* (max(mrt)-min(mrt)) / 5
        else:
            minn=0
            maxi=10 #Not used

        for i in np.random.randint(minn,maxi,args.time_per_mrt):
            kw={}

            param_k = ["maxv","minv","pulselen","inct","dect"]

            for p in param_k:
                kw[p] = draw(params[p])

            if args.correct_for_height:

                kw["maxv"] = kw["maxv"]/(1-np.exp(-2/kw["inct"]))

            if not args.one_fork:
                tc,len_initial,kw = generate_track(sim,start_time=i,
                                                   end=chlen,params=kw)

                rfds = generate_rfd(sim,end=chlen)
            else:

                time= np.arange(0,chlen/average_fork_speed,1/average_fork_speed)
                #start at 1kb
                tc,len_init,_ = track(time,start_time=time[1000//resolution-1],
                                    end_time=chlen/average_fork_speed,**kw)
                rfds=None
                kw["speed"]=len_init  / kw["pulselen"] * resolution

            for size in np.random.choice(possiblesize,p=distribsize,size=args.read_per_time):

                start = np.random.randint(0,len(tc)-size)
                if whole_length:
                    start=0
                    size = len(tc)-1

                attemp=0
                while (rfds is not None and np.sum(rfds[start:start+size]) == 0) or  ((np.max(tc[start:start+size]) - np.min(tc[start:start+size])) < 0.3) :
                    start = np.random.randint(0,len(tc)-size)
                    attemp += 1
                    if attemp > 100:
                        break

                if not args.one_fork:
                    # Get speeds of non nul forks
                    kw["speed"]=[li / kw["pulselen"] * resolution\
                                     for li,[startf,endf] in zip(len_initial[0],
                                                               len_initial[1]) \
                                  if (li != 0) and startf>start and endf<start+size ]

                ui = str(uuid.uuid4())
                gt[ui] = tc[start:start+size]
                f = tc[start:start+size].copy()

                val_background =  stats.lognorm.rvs(s=1,scale=0.017*1.48,loc=0.015, size=1)[0]
                while val_background>0.2:
                    val_background =  stats.lognorm.rvs(s=1,scale=0.017*1.48,loc=0.015, size=1)[0]

                f+=val_background
                f[f>1]=1
                kw["val_background"]=val_background

                n_info = np.random.randint(15,60)
                f= np.random.binomial(n_info,f) / n_info
                f = smooth(f,2)

                fiber[ui] = f
                kw["speed_th"] = average_fork_speed * resolution
                parameters[ui] = kw
                pos[ui] = np.arange(start,start+size)

                if args.one_fork:
                    positions[ui]=[1000//resolution,1000//resolution+len_init,1]
                    all_speeds[ui]=[kw.pop("speed")]
                else:
                    all_speeds[ui]=kw.pop("speed")
                    positions[ui]=[[startf-start,endf-start,(-1)**(posn+1)] for posn,(li,[startf,endf]) in enumerate(zip(len_initial[0],
                                                               len_initial[1])) \
                                  if (li != 0) and startf>start and endf<start+size ]
                if not args.one_fork and rfds is not None:
                    rfd[ui] = rfds[start:start+size]

    k = list(fiber.keys())
    print(len(fiber.keys()),"len")
    kp = np.random.permutation(len(k))
    permuted = np.array(k)[kp]


    if args.ground_truth:
        with open(f"{args.prefix}_gt.fa","w") as h:
            for p in permuted:
                formated = ["%.2f"%v for v in gt[p]]
                h.writelines(f"{p}\n {' '.join(formated)}\n")
    if args.fork_position:
        with open(f"{args.prefix}_positions.fa","w") as h:
            for p in permuted:
                formated = [str(v) for v in positions[p]]
                h.writelines(f"{p}\n {str(positions[p])}\n")


    with open(f"{args.prefix}.fa","w") as f, \
         open(f"{args.prefix}_parameters.txt","w") as g, \
         open(f"{args.prefix}_all_speeds.txt","w") as j:
        for p in permuted:
            g.writelines(f"{p} {str(parameters[p])}\n")
            formated = ["%.2f"%v for v in fiber[p]]
            f.writelines(f"{p}\n {' '.join(formated)}\n")
            formated = ["%.2f"%v for v in all_speeds[p]]
            j.writelines(f"{p}\n {' '.join(formated)}\n")

    if args.draw_sample != 0:
        n=20
        f = pylab.figure(figsize=(20,20))
        k = list(fiber.keys())
        kp = np.random.permutation(len(k))
        #k1 = list(speed.keys())
        #print(k[:10],k1[:10])
        for i in range(args.draw_sample-1):
            f.add_subplot(n//2,2,i+1)
            ui = np.array(k)[kp][i]
            ftp  = fiber[ui]
            pylab.plot(np.arange(len(ftp))/10,ftp,label=parameters[ui]["maxv"])
            #plot(np.arange(len(ftp))/10,rfd[np.array(k)[kp][i]])
            pylab.ylim(0,1.1)
            #xlim(0,50)
            pylab.xlabel("kb")
            pylab.ylabel("% Brdu")
            pylab.legend()
        f.tight_layout()
        pylab.savefig(f"{args.prefix}_sample.pdf")
