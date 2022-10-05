
import numpy as np
from collections import namedtuple
from pomegranate import GeneralMixtureModel,NormalDistribution
import pandas as pd
def smooth(ser, sc):
    return np.array(pd.Series(ser).rolling(sc, min_periods=1, center=True).mean())


origin = namedtuple("origin",["pos","firing_time","L_fork_speed","R_fork_speed"])
Pause = namedtuple("pause",["pos","duration"])


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

def intersection(p1,p2,pause=[0,0]):
    """
    Given two converging forks and their firing time and speeds,
    compute the position of the intersection
    as well as the position of the time of intersection.
    If the intersection is outside [x1,x2], the initial position of the forks,
    then return False
    """
    x1,t1,R_fork_speed=p1.pos,p1.firing_time,p1.R_fork_speed
    x2,t2,L_fork_speed=p2.pos,p2.firing_time,p2.L_fork_speed
    t1 += pause[0]
    t2 += pause[1]

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

def generate_track(pos_time,start_time=10,end=1000,params={},same_parameters=True,pauses=[]):
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


    if len(pauses) ==0:
        pauses=[Pause(pos=None,duration=0)] * (len(pos_time)+1)
    #CHeck that pauses are ordered
    if len(pauses)>1:
        for p1,p2 in zip(pauses[:-1],pauses[1:]):
            if p1.pos != None and p2.pos != None:
                assert(p1.pos<p2.pos)

    #insert empty pauses and order pause. check only one pause between all ori
    #print("Before",pauses)
    #print(pauses)
    if len(pauses) != (len(pos_time)+1):
        f_pauses=[None]*(len(pos_time)+1)
        p_ori_order=[ori.pos for ori in pos_time]
        #print(p_ori_order)
        startp=0
        if pauses[0].pos<p_ori_order[0]:
            f_pauses[0]=pauses[0]
            startp=1

        for pause in pauses[startp:]:
            for ipos in range(len(p_ori_order)-1):
                if p_ori_order[ipos+1]>pause.pos>=p_ori_order[ipos]:

                    if f_pauses[ipos+1] != None:
                        print("At least two pauses located between two origins")
                        print("Origins",pos_ori)
                        print("Pauses",pauses)
                        raise
                    else:
                        f_pauses[ipos+1]=pause
        if pauses[-1].pos>p_ori_order[-1]:
            f_pauses[-1]=pauses[-1]
        for i in range(len(f_pauses)):
            if f_pauses[i] == None:
                f_pauses[i]=Pause(pos=None,duration=0)
        #print("After",f_pauses)
        pauses=f_pauses
    else:
        #Pauses must be located between origins
        for pause,ori in zip(pauses,pos_time):
            if pause.pos != None:
                assert(pause.pos<=ori.pos)
        for pause,ori in zip(pauses[1:],pos_time[:]):
            if pause.pos != None:
                assert(pause.pos>=ori.pos)



    assert(len(pauses)==len(pos_time)+1)

    #def generate_time(start_t,pos_end,speed,pause=0):
    #    return np.arange(start_t,start_t+pos_end/speed,1/speed)


    trac = np.zeros(end)

    x1,t1,L_fork_speed = pos_time[0].pos,pos_time[0].firing_time,pos_time[0].L_fork_speed
    time= np.arange(t1,t1+x1/L_fork_speed,1/L_fork_speed)
    if pauses[0].duration != 0:
        #print(pauses)
        time[x1-pauses[0].pos:]+=pauses[0].duration

    t,len_init,before = track(time,start_time=start_time,end_time=t1+x1/L_fork_speed+pauses[0].duration,
                               **generate_params(param_k,kw))
    trac[:x1] = t[:x1][::-1]

    #print(len_init)
    mrt = [time[:x1][::-1]]
    #mrt[:x1] = time[:x1][::-1]
    len_initial = [len_init + 0] #store the length of the increasing parts
    pos_s = [[x1-len_init-before,x1-before]]
    for interval,(p1,p2,pause) in enumerate(zip(pos_time[:-1],pos_time[1:],pauses[1:]),1):
        if pause.duration !=0:
            assert(p2.pos>pause.pos>p1.pos)
        possible,inte = intersection(p1,p2)
        middle = int(round(inte[0]))
        first_encounter_pause=True
        if pause.duration!=0:
            if middle > pause.pos:
                #First fork get their first
                delta=middle-pause.pos
                delta_t=delta/p2.L_fork_speed
                if delta_t>pause.duration:
                    #Then fork1 finish its pause
                    #Equivalent to starting late of time pause
                    possible,inte = intersection(p1,p2,pause=[pause.duration,0])
                    middle = int(round(inte[0]))
                else:
                    pauses[interval] = Pause(pos=pause.pos,duration=delta/p2.L_fork_speed)
                    pause=pauses[interval]
                    middle = pause.pos
            else:
                first_encounter_pause = False
                delta=pause.pos-middle
                delta_t=delta/p1.L_fork_speed
                if delta_t >pause.duration:
                    #Then fork2 finish its pause
                    possible,inte = intersection(p1,p2,pause=[0,pause.duration])
                    middle = int(round(inte[0]))
                else:
                    pauses[interval] = Pause(pos=pause.pos,duration=delta/p1.L_fork_speed)
                    pause=pauses[interval]

                    middle = pause.pos


        size = len(trac[p1[0]:middle])
        starto = p1.firing_time
        R_fork_speed = p1.R_fork_speed


        time= np.arange(starto,starto+size/R_fork_speed,1/R_fork_speed)
        end_cover=0
        if pause.duration != 0:
            end_cover=pause.duration
        if first_encounter_pause and pause.duration !=0:
            time[pause.pos-p1.pos:] += pause.duration

        mrt.append(time[:size])
        #print(time)
        #print(time,len(time))
        #print(p1[0],p2[0],middle)
        #print(track(time,start_time=start_time,end_time=starto+size/v)[:size])
        #trac[p1.pos:middle]
        t,len_init,before= track(time,start_time=start_time,end_time=starto+size/R_fork_speed+end_cover,
                                   **generate_params(param_k,kw))
        trac[p1.pos:middle] = t[:size]
        len_initial.append(len_init + 0)
        pos_s += [[p1.pos+before,p1.pos+len_init+before]]

        size = len(trac[middle:p2.pos])
        starto = p2.firing_time
        L_fork_speed = p2.L_fork_speed

        time= np.arange(starto,starto+size/L_fork_speed,1/L_fork_speed)
        if not first_encounter_pause and pause.duration !=0:
            time[p2.pos-pause.pos:] += pause.duration


        mrt.append(time[:size][::-1])
        #print(time,len(time))
        trac[middle:p2.pos]

        t,len_init,before = track(time,start_time=start_time,end_time=starto+size/L_fork_speed+end_cover,
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

    if pauses[-1].duration != 0:
            #print(pauses)
        time[pauses[-1].pos-x2:]+=pauses[-1].duration

    mrt.append(time[:size])
    #mrt[x2:] = time[:size]


    t,len_init,before = track(time,start_time=start_time,end_time=t2+size/R_fork_speed+pauses[-1].duration,
                               **generate_params(param_k,kw))
    trac[x2:] = t[:size]
    len_initial.append(len_init + 0)
    pos_s += [[x2+before,x2+len_init+before]]

    if not same_parameters:
        kw = list_param_generated
    #print(len(trac),len(np.concatenate(mrt)))
    #print(pauses,R_fork_speed)
    return trac,[len_initial,pos_s],kw,mrt


def create_possible_origins(n_ori,n_sim,average_fork_speed,chlen,scaling=15):
    """
    generate a list of possible simulation given a number of origin an average_fork_speed
    """
    sims =[]
    #print(chlen,n_ori,chlen/n_ori/scaling,scaling)
    while len(sims) != n_sim:
        #print(len(sims))
        pos = np.random.randint(0,chlen,n_ori)
        times = np.random.randint(0,chlen/n_ori/scaling,n_ori)
        #times=np.ones_like(times)
        times -= min(times)
        pos.sort()
        if len(set(pos)) != len(pos):
            continue
        #print(pos)
        sim = [origin(p,t,average_fork_speed,average_fork_speed) for p,t in zip(pos,times)]
        #print(sim)
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
    import ast

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
    parser.add_argument('--conf',type=str,default=None,help="configuration of origins to simulate from")

    parser.add_argument('--test',  action="store_true")
    parser.add_argument('--whole_length',  action="store_true")
    parser.add_argument('--length',  type=int,default=None)
    parser.add_argument('--bckg',  type=str,default="repnano",choices=["repnano","meg3","meg3_res1"])
    parser.add_argument('--no_mrt', dest="mrt",action="store_false")
    parser.add_argument('--seed',  type=int,default=None)
    parser.add_argument('--correlation',action="store_true")
    parser.add_argument('--rfd',action="store_true")
    parser.add_argument('--zeros',action="store_true")
    parser.add_argument('--states',action="store_true")



    args = parser.parse_args()

    if args.seed != None:
        np.random.seed(args.seed)


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

    if args.test:
        chlen=50000 //resolution
        whole_length=True

    if args.whole_length:
        whole_length=True
    if args.length != None:
        chlen=int(args.length/resolution)

    possiblesize = np.arange(5000//resolution,chlen)
    distribsize = stats.lognorm(0.5,scale=35000/resolution).pdf(possiblesize)
    distribsize /= np.sum(distribsize)

    nfork = {}
    pos={}
    fiber = {}
    rfd = {}
    mrts = {}
    starts={}
    ends={}
    start_times = {}
    deltas=[]

    gt = {}
    parameters = {}
    all_speeds={}
    positions = {}
    pulse_lens={}
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

    if args.correlation:
        import pickle
        fich=params["correlated"]["file"]
        with open(fich,"rb") as f:
            correlated_sample=pickle.load(f)


    if args.conf != None:
        Confs = []
        Pauses = []
        with open(args.conf,"r") as f:
            for line  in f.readlines():
                new_conf = ast.literal_eval(line)
                average_fork_speed = draw(params["speed"]) / resolution
                ori_pos =[]
                for ori in new_conf[0]:
                    if len(ori)==4:
                        ori[0] = int(ori[0]/resolution)
                        ori_pos.append(origin(*ori))
                    elif len(ori)==2:
                        ori[0] /=resolution
                        ori_pos.append(origin(int(ori[0]),ori[1],average_fork_speed,average_fork_speed))
                    else:
                        raise
                Confs.append(ori_pos)

                if len(new_conf)==2:
                    pt = []
                    for p in new_conf[1]:
                        p[0]/=resolution
                        pt.append(Pause(int(p[0]),p[1]))
                    Pauses.append(pt)
        n_conf=len(Confs)
        #print(Confs)
        #print(Pauses)

    else:
        n_conf = args.n_conf_ori

    for sim_number  in range(n_conf): # [current]:
        if sim_number % 500 == 0:
            print(sim_number,sim_number/n_conf)

        average_fork_speed = draw(params["speed"]) / resolution

        if average_fork_speed<=0:
            continue
        pauses=[]
        if not args.one_fork:

            if args.test:
                sim=[origin(100,0,average_fork_speed,average_fork_speed)]
                #pauses=[Pause(pos=49,duration=20),Pause(pos=120,duration=4)]

                sim=[origin(30,0,average_fork_speed,average_fork_speed),origin(150,0,average_fork_speed,average_fork_speed)]
                pauses=[]
                #pauses=[Pause(pos=140,duration=10)]
                #pauses=[Pause(pos=0,duration=0),Pause(pos=140,duration=10),Pause(pos=180,duration=0)]
                #pauses=[Pause(pos=140,duration=10)]
                #pauses=[Pause(pos=0,duration=0),Pause(pos=120,duration=0.5),Pause(pos=180,duration=0)]
                #pauses=[Pause(pos=0,duration=0),Pause(pos=80,duration=10),Pause(pos=180,duration=0)]
                #pauses=[Pause(pos=0,duration=0),Pause(pos=80,duration=0.5),Pause(pos=180,duration=0)]


            elif args.conf != None:
                sim=Confs[sim_number]
                pauses=Pauses[sim_number]
                average_fork_speed = np.mean(np.concatenate([[ori.L_fork_speed,ori.R_fork_speed] for ori in ori_pos]))
            else:
                sim = create_possible_origins(int(chlen / (args.average_distance_between_ori / resolution)),1,
                                      average_fork_speed,chlen,scaling=15*100/args.resolution)[0]

                if len(sim)>1:
                    #print(sim)
                    deltas.extend([ori2.pos-ori1.pos for ori1,ori2 in zip(sim[:-1],sim[1:])])
                else:
                    deltas.append(np.nan)

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

            if args.correlation:
                kw["pulselen"] = draw(params["pulselen"])
                correlated_sample_val = correlated_sample[np.random.randint(len(correlated_sample))]
                for ite_param,val in enumerate(params["correlated"]["params"]):
                    kw[val]=correlated_sample_val[ite_param]
            else:
                for p in param_k:
                    kw[p] = draw(params[p])

            if args.correct_for_height:

                kw["maxv"] = kw["maxv"]/(1-np.exp(-2/kw["inct"]))

            if not args.one_fork:
                tc,len_initial,kw,mrt = generate_track(sim,start_time=i,
                                                   end=chlen,params=kw,pauses=pauses)

                rfds = generate_rfd(sim,end=chlen)
                start_time = i
            else:

                time= np.arange(0,chlen/average_fork_speed,1/average_fork_speed)
                #start at 1kb
                start_mono=10000
                tc,len_init,_ = track(time,start_time=time[start_mono//resolution-1],
                                    end_time=chlen/average_fork_speed,**kw)

                start_time = time[start_mono//resolution-1]
                kw["speed"]=len_init  / kw["pulselen"] * resolution
                mrt=time
                rfds = np.ones(len(tc))
            pulselen = kw["pulselen"]
            #print(kw)
            for size in np.random.choice(possiblesize,p=distribsize,size=args.read_per_time):

                start = np.random.randint(0,len(tc)-size)
                if whole_length:
                    start=0
                    size = len(tc)
                else:
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
                if args.ground_truth:
                    gt[ui] = np.array(tc[start:start+size],dtype=np.float16)
                f = tc[start:start+size].copy()

                if args.bckg == "repnano":
                    val_background =  stats.lognorm.rvs(s=1,scale=0.017*1.48,loc=0.015, size=1)[0]
                    while val_background>0.2:
                        val_background =  stats.lognorm.rvs(s=1,scale=0.017*1.48,loc=0.015, size=1)[0]

                    f+=val_background
                    f[f>1]=1
                    kw["val_background"]=val_background

                    n_info = np.random.randint(15,60)
                    f= np.random.binomial(n_info,f) / n_info
                    f = smooth(f,2)
                elif args.bckg == "meg3":
                    lgparam=[1.978551381625039, -4.0862276995991705e-06, 0.001]
                    val_background =  stats.lognorm.rvs(*lgparam, size=1)[0]
                    while val_background>0.2 or val_background<1e-7:
                        val_background =  stats.lognorm.rvs(*lgparam, size=1)[0]
                    #print(val_background)
                    f+=val_background
                    f[f>1]=1
                    kw["val_background"]=val_background

                    n_info = np.random.randint(10,60,len(f))
                    f= np.random.binomial(n_info,f) / n_info

                elif args.bckg == "meg3_res1":
                    lgparam=[1.978551381625039, -4.0862276995991705e-06, 0.001]
                    val_background =  stats.lognorm.rvs(*lgparam, size=1)[0]
                    while val_background>0.2 or val_background<1e-7:
                        val_background =  stats.lognorm.rvs(*lgparam, size=1)[0]
                    #print(val_background)
                    #gaussn = np.random.normal(0,scale=0.8,size=len(f))
                    #f[f!=0]+=gaussn[f!=0]
                    f[f<0]=0
                    f+=val_background

                    f[f>1]=1
                    kw["val_background"]=val_background
                    #n_info = np.random.randint(10,60,len(f))
                    #f= np.array(np.random.binomial(1,f),dtype=np.int16)
                    n_info = np.random.randint(1,3,len(f))
                    f= np.random.binomial(n_info,f) / n_info


                    nan=np.random.randint(0,len(f),size=int(1.5*len(f)))
                    #print(f[9500:10500])
                    #print(nan)
                    #print(len(nan))

                    #f = np.array(f,dtype=bool)
                    f[nan]=-1
                    #f=np.array(f,flo)
                    #print(np.sum(np.isnan(f))/len(f),len(f),np.sum(np.isnan(f)))
                        #f = smooth(f,2)

                else:
                    print(f"bacground {args.bckg} not implemented")
                    raise


                fiber[ui] = f
                kw["speed_th"] = average_fork_speed * resolution
                parameters[ui] = kw
                starts[ui]=starts
                ends[ui]=ends
                #pos[ui] = np.arange(start,start+size)
                if args.mrt or args.zeros or args.states:
                    mrts[ui]=np.array(np.concatenate(mrt)[start:start+size],dtype=np.float16)
                    start_times[ui] = start_time


                if args.one_fork:
                    positions[ui]=[start_mono//resolution,start_mono//resolution+len_init,1]
                    all_speeds[ui]=[kw["speed"]]
                else:
                    all_speeds[ui]=kw["speed"]
                    positions[ui]=[[startf-start,endf-start,(-1)**(posn+1)] for posn,(li,[startf,endf]) in enumerate(zip(len_initial[0],
                                                               len_initial[1])) \
                                  if (li != 0) and startf>start and endf<start+size ]
                #if not args.one_fork and rfds is not None:
                rfd[ui] = rfds[start:start+size]
                pulse_lens[ui] = pulselen


    k = list(fiber.keys())
    print(len(fiber.keys()),"len")
    if args.conf !=None:
        permuted = np.array(k)
    else:
        kp = np.random.permutation(len(k))
        permuted = np.array(k)[kp]

    if deltas != []:
        print(f"Average distance {np.nanmean(deltas)}, fraction of mono ori {np.mean(np.isnan(deltas))}")
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
    if args.rfd:
        with open(f"{args.prefix}_rfds.fa","w") as f:
            for p in permuted:

                formated = ["%i"%v  for v in rfd[p]]
                f.writelines(f"{p}\n {' '.join(formated)}\n")

    if args.zeros:
        with open(f"{args.prefix}_zeros.fa","w") as f:
            for p in permuted:
                start_t = start_times[p]
                formated = ["0" if v> start_t else "1"  for v in mrts[p]]
                f.writelines(f"{p}\n {' '.join(formated)}\n")

    if args.states:
        with open(f"{args.prefix}_states.fa","w") as f:
            for p in permuted:
                start_t = start_times[p]
                pulse_len = pulse_lens[p]
                def which_state(mrt,rfd,start_t,pulse_len):
                    if mrt < start_t:
                        return "0"
                    else:
                        if start_t<mrt<start_t + pulse_len:
                            return str(rfd)
                        else:
                            return str(2*rfd)



                formated = [which_state(mrti,rfdi,start_t,pulse_len)  for mrti,rfdi in zip(mrts[p],rfd[p])]
                f.writelines(f"{p}\n {' '.join(formated)}\n")


    with open(f"{args.prefix}.fa","w") as f, \
         open(f"{args.prefix}_parameters.txt","w") as g, \
         open(f"{args.prefix}_all_speeds.txt","w") as j:
        for p in permuted:
            g.writelines(f"{p} {str(parameters[p])}\n")
            if args.resolution != 1:
                formated = ["%.2f"%v for v in fiber[p]]
            else:
                formated = ["%.2f"%v if v != -1 else "%.2f"%np.nan  for v in fiber[p]]
            f.writelines(f"{p}\n {' '.join(formated)}\n")
            formated = ["%.2f"%v for v in all_speeds[p]]
            j.writelines(f"{p}\n {' '.join(formated)}\n")



    if args.draw_sample != 0:

        maxi = min(args.draw_sample,len(k))
        f = pylab.figure(figsize=(20,maxi))
        #k1 = list(speed.keys())
        #print(k[:10],k1[:10])
        for i in range(maxi):
            f.add_subplot(maxi//2,2,i+1)
            ui = np.array(permuted)[i]
            ftp  = fiber[ui]
            if args.resolution == 1:
                ftp =np.array(ftp,dtype=float)
                ftp[ftp==-1]=np.nan
            #print(len(mrts[ui]))
            #print(mrts[ui],type(mrts[ui]))
            if ui in mrts:
                if len(mrts[ui])>1 and (type(mrts[ui]) ==list):
                    mrt=np.concatenate(mrts[ui])
                else:
                    mrt=mrts[ui].flatten()

            pylab.plot(np.arange(len(ftp))*resolution/1000,ftp,"-o",label=parameters[ui]["maxv"])
            if args.resolution == 1:
                pylab.plot(np.arange(len(ftp))*resolution/1000,smooth(smooth(ftp,100),2500),label=parameters[ui]["maxv"])

            #pylab.plot(np.arange(len(ftp))/10,gt[ui],label=parameters[ui]["maxv"])
            if not args.resolution==1 and ui in mrt:
                pylab.plot(np.arange(len(mrt))*resolution/1000,mrt/max(mrt),label=parameters[ui]["maxv"])

            #plot(np.arange(len(ftp))/10,rfd[np.array(k)[kp][i]])
            pylab.ylim(0,1.1)
            #xlim(0,50)
            pylab.xlabel("kb")
            pylab.ylabel("% Brdu")
            pylab.legend()
        f.tight_layout()
        pylab.savefig(f"{args.prefix}_sample.pdf")
