import numpy as np
import matplotlib.pyplot as plt
import pickle
import tkinter
import threading
import argparse
import conf
import time
import math

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

def boundv(v):
    if (v > conf.vHIGH): return conf.vHIGH
    elif (v < conf.vLOW): return conf.vLOW
    else: return v

def boundx(x):
    if (x > conf.xHIGH): return conf.xHIGH
    elif (x < conf.xLOW): return conf.xLOW
    else: return x

# environment function returning reward and state from state and action
# mountain car
def env(xv,a):
    x = xv[0]
    v = xv[1]
    vnew = boundv(v+0.001*a-0.0025*math.cos(3*x))
    xnew = boundx(x+vnew)
    if (xnew <conf.xLOW): vnew = 0
    reward = -1
    if (xnew >= conf.xHIGH): reward = 0
    return reward,[xnew,vnew]


class compute(threading.Thread):

    def __init__(self):
        super(compute,self).__init__()
        self.policy = 0
        self.S = [[x,v] for x in range(0,conf.nx) for v in range(0,conf.nv)]
        self.A = [-1,0,1]
        self.cSA = [self.SA_to_cSA([state,action]) for state in self.S for action in self.A]
        self.cSA_to_iSA = {}
        for i in range(0,len(self.cSA)): self.cSA_to_iSA[self.cSA[i]] = i
        self.iSA_to_cSA = {}
        for i in range(0,len(self.cSA)): self.iSA_to_cSA[i] = self.cSA[i]

        # current trajectory
        self.trajx = np.array([])
        self.trajv = np.array([])
        
        self.weights = [0 for i in range(0,len(self.cSA))]
    def loadQhat(self,file):
        with open(file, 'rb') as f:
            data = pickle.loads(f.read())
            self.policy = data['policy']
            self.weights = data['weights']
            f.close()
        print('loaded policy file from: ',file)

    def saveQhat(self,file):
        with open(file, 'wb') as f:
            pickle.dump({'policy':self.policy,'weights':self.weights}, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        print('policy file saved to: ',file)

# load configure parameters passed by the argparse
    def setConfigure(self,args):
        self.args = args

    def setModel(self,model):
        #self.model = {'sarsa':self.sarsa,'sarsaconv':self.sarsaconv,'sarsalambda':self.sarsalambda,'Q':self.Qlearn}[model]
        self.model = self.sarsa

    def setRenderer(self,renderer):
        self.renderer = renderer

    def setStop(self,stopevent):
        self.stopevent = stopevent

    # find policy globally for Q-value function
    def greedy(self,epsilon):

        policy = {}
        for state in self.S:
            Qpoint = [self.Qhat(self.SA_to_vSA([state,action])) for action in self.A]
            #print(Qpoint)
            maxaction_indices = np.argwhere(Qpoint == np.amax(Qpoint)).flatten().tolist()
            #print(maxaction_indices)
            prob = [epsilon/len(Qpoint)  for action in self.A]
            for i in maxaction_indices: prob[i] = prob[i] + (1-epsilon)/len(maxaction_indices)
            policy[self.S_to_cS(state)] = prob
        return policy


    # choose a move for a state according to policy
    def choosemove(self,cS):
        return np.random.choice(self.A,p=self.policy[cS])

    # S - state
    # cS - coded state
    # cSA - coded action-state
    # vSA - vector action-state

    def xv_to_S(self,xv):
        offset=0.0001 # offset
        nx=conf.nx
        nv=conf.nv
        xtilesize = (conf.xHIGH-conf.xLOW)/nx
        vtilesize = (conf.vHIGH-conf.vLOW)/nv
        x = xv[0]
        v = xv[1]
        xtile = int((x-conf.xLOW)/xtilesize-offset)
        vtile = int((v-conf.vLOW)/vtilesize-offset)
        return [xtile,vtile]

    def xv_to_cS(self,xv):
        return self.S_to_cS(self.xv_to_S(xv))

    # coding and decoding functions
    def SA_to_cSA(self,stateaction):
        return str(stateaction[0][0])+"_"+str(stateaction[0][1])+"_"+str(stateaction[1])

    def cSA_to_SA(self,stateaction):
        split = stateaction.split("_")
        return [[int(split[0]),int(split[1])],int(split[-1])]

    def S_to_cS(self,state):
        return str(state[0])+"_"+str(state[1])

    def cS_to_S(self,statecoded):
        split = statecoded.split("_")
        return [int(split[0]),int(split[1])]

    def SA_to_vSA(self,SA):
        return self.cSA_to_vSA(self.SA_to_cSA(SA))

    def cSA_to_vSA(self,cSA):
        ind = self.cSA_to_iSA[cSA]
        vSA = [0 for i in range(0,len(self.cSA))]
        vSA[ind] = 1
        return vSA

    def SA_to_iSA(self,SA):
        return self.cSA_to_iSA[self.SA_to_cSA(SA)]

    # Q function approximator
    def Qhat(self,vSA):
        return np.inner(self.weights,vSA)

    def collecttrajectory(self,data):
        self.trajx = np.append(self.trajx,data[0])
        self.trajv = np.append(self.trajv,data[1])

    def cleartrajectory(self):
        self.trajx = np.array([])
        self.trajv = np.array([])

    def simulate(self):

        n=args.noruns
        i=1
        while n != 0:
            if (type(args.xvinit)==list): xv = [args.xvinit[0],args.xvinit[1]]
            else: xv = [np.random.uniform(conf.xLOW,conf.xHIGH),np.random.uniform(conf.vLOW,conf.vHIGH)]

            S = self.xv_to_S(xv)
            A = self.choosemove(self.S_to_cS(S))
            #print(self.policy)
            print(i,'; init [x,v]=',xv)
            t = 0
            while not xv[0] == conf.xHIGH:
                self.collecttrajectory(xv)
                time.sleep(args.lag)
                #print(xv)
                if self.stopevent.isSet(): return None
                if not self.args.fast: self.renderer.movecar(xv)
                R,xv_prime = env(xv,A)
                S_prime = self.xv_to_S(xv_prime)
                A_prime = self.choosemove(self.S_to_cS(S_prime))
                xv = xv_prime
                S = self.xv_to_S(xv)
                A = A_prime
                t+=1
            print('reached the goal after t=%s' % t)
            self.renderer.addsuccess(t)
            self.cleartrajectory()
            n-=1
            i+=1

    def sarsa(self):
        epsilon = conf.defaultepsilon
        alpha = conf.defaultalpha
        gamma = conf.defaultgamma
        time_th = conf.timethreshold if self.args.reset else -1 # set time threshold if the trajectory get stuck
        #noepisodes = conf.defaultnoepisodes

        self.weights = [0 for i in range(0,len(self.cSA))]
        self.policy = self.greedy(epsilon)

        print('running SARSA with epsilon=%s,alpha=%s,gamma=%s' % (epsilon,alpha,gamma))
        n=args.noruns
        j=1
        while n != 0:
            xv = [np.random.uniform(conf.xLOW,conf.xHIGH),np.random.uniform(conf.vLOW,conf.vHIGH)]
            #xv = [0,0]
            S = self.xv_to_S(xv)
            A = self.choosemove(self.S_to_cS(S))
            self.policy = self.greedy(epsilon)

            print(j,'; init [x,v]=',xv)
            visits = [1 for i in range(0,len(self.cSA))]
            propagation = 0
            countdown = time_th
            time0=0
            while not (xv[0] == conf.xHIGH or countdown == 0 ):
                self.collecttrajectory(xv)
                #print(xv)
                #print(len(self.trajx),len(self.trajv))
                #if propagation<len(self.trajx): propagation=len(self.trajx)
                #else: input()
                #print(visits)
                #input()
                #visits[self.SA_to_iSA([S,A])]+=1
                if self.stopevent.isSet(): return None
                if not self.args.fast: self.renderer.movecar(xv)
                R,xv_prime = env(xv,A)
                S_prime = self.xv_to_S(xv_prime)
                self.policy = self.greedy(epsilon)
                A_prime = self.choosemove(self.S_to_cS(S_prime))
                self.weights[self.SA_to_iSA([S,A])] += alpha/visits[self.SA_to_iSA([S,A])]*(R+ gamma*self.Qhat(self.SA_to_vSA([S_prime,A_prime])) - self.Qhat(self.SA_to_vSA([S,A])))
                xv = xv_prime
                S = self.xv_to_S(xv)
                A = A_prime
                countdown-=1
                time0+=1
            if (countdown==0):
                print('trajectory got stuck after conf.timethreshold=%s iterations, moving on' % time_th)
            else:
                print('reached the goal after t=%s' % time0)
            self.renderer.addsuccess(time0)
            self.cleartrajectory()
            n-=1
            j+=1

    def run(self):
        if self.args.mode == 'r':
            self.loadQhat(self.args.path)
            self.simulate()
        elif self.args.mode == 'f':
            self.model()
        elif self.args.mode == 'c':
            self.loadQhat(self.args.path)
            self.model()
            #self.saveQhat(self.args.path)


class render():

    def __init__(self,root):
        self.root = root

        self.mainFrame = tkinter.Frame(self.root,bg="white")
        self.mainFrame.pack(side="top") # position the mainFrame

        # simulation frame
        self.simFrame = tkinter.Frame(self.mainFrame)

        # myCanvas + buttonFrame -> simFrame
        self.myCanvas = tkinter.Canvas(self.simFrame, bg="white", height=conf.scaley, width=conf.scalex)
        self.myCanvas.xview_scroll(300,"pages")
        self.myCanvas.yview_scroll(300,"pages")
        self.myCanvas.pack(side="top")

        self.buttonFrame = tkinter.Frame(self.simFrame, bg="white")
        self.buttonSave = tkinter.Button(self.buttonFrame,text="save current policy",command=self.savepolicy,state={'r':'disabled','f':'normal','c':'normal'}[args.mode])
        self.buttonSave.pack(side="left")
        self.buttonFrame.pack(side="top")

        # success time plot
        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_ylim(0,200)
        self.ax3.set_xlim(0,50)

        self.plt3 = self.ax3.plot([],[])
        self.ax3.set_xlabel('#')
        self.ax3.set_ylabel('success time')
        self.line3, = self.plt3

        self.figCanvas3 = FigureCanvasTkAgg(self.fig3,master=self.mainFrame)

        # phase space
        self.fig = Figure()
        ax = self.fig.add_subplot(111)

        ax.set_ylim(conf.vLOW,conf.vHIGH)
        ax.set_xlim(conf.xLOW,conf.xHIGH)

        # draw gridlines to match with the state space
        dv = (conf.vHIGH-conf.vLOW)/conf.nv
        dx = (conf.xHIGH-conf.xLOW)/conf.nx
        vlines = [conf.xLOW + (i)*dx for i in range(conf.nx)]
        hlines = [conf.vLOW + (i)*dv for i in range(conf.nv)]
        for vline in vlines:
            ax.axvline(x=vline,ls='--')
        for hline in hlines:
            ax.axhline(y=hline,ls='--')

        ax.set_xlabel('position x')
        ax.set_ylabel('velocity v')
        self.plt = ax.plot([],[])

        self.line, = self.plt

        self.figCanvas = FigureCanvasTkAgg(self.fig,master=self.mainFrame)


        # weight plot
        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.ax2.set_ylim(-5,5)
        self.ax2.set_xlim(0,108)

        self.plt2 = self.ax2.plot([],[])
        self.ax2.set_xlabel('state-action')
        self.ax2.set_ylabel('weight')
        self.line2, = self.plt2

        self.figCanvas2 = FigureCanvasTkAgg(self.fig2,master=self.mainFrame)


        # positioning of plots

        self.simFrame.grid(column=0,row=0)
        self.figCanvas.get_tk_widget().grid(column=0,row=1) # phase space plot
        self.figCanvas3.get_tk_widget().grid(column=1,row=0) # success time plot
        self.figCanvas2.get_tk_widget().grid(column=1,row=1) # weight plot


    def setConfigure(self,args):
        self.args = args

    def setComputer(self,computer):
        self.computer = computer

    def savepolicy(self):
        self.computer.saveQhat(self.args.path)

    def drawground(self):
        imax = 50
        xs = [conf.xLOW+i/imax*(conf.xHIGH-conf.xLOW) for i in range(0,imax+1)]
        ys = list(map(lambda x: -0.5*math.sin(3*x),xs))
        for i in range(0,len(xs)-1): self.myCanvas.create_line(conf.scale*xs[i],conf.scale*ys[i],conf.scale*xs[i+1],conf.scale*ys[i+1],width=2,fill="black")

    def drawscene(self):
        self.myCanvas.create_line(conf.xLOW*conf.scale,-100,conf.xLOW*conf.scale,100,width=2,fill="black")
        self.myCanvas.create_line(conf.xHIGH*conf.scale,-100,conf.xHIGH*conf.scale,100,width=2,fill="black")

        self.drawground()
        #self.id = self.myCanvas.create_rectangle(conf.xLOW*conf.scale,0,conf.xHIGH*conf.scale,10,width=.1,fill="red")
        self.drawcar()

    def drawcar(self):
        self.car = self.myCanvas.create_oval(-10,-10,+10,+10,fill="black")
        #print(self.myCanvas.coords(self.car))

    def drawtrajectory(self):
        self.line.set_ydata(self.computer.trajv)
        self.line.set_xdata(self.computer.trajx)
        self.figCanvas.draw()

    def drawweights(self):
        if self.computer.weights:
            mean = np.mean(self.computer.weights)
            std = np.std(self.computer.weights)
            title = 'weight mean='+str(mean)[:7]+'; std='+str(std)[:7]
            self.ax2.set_title(title)
            self.line2.set_ydata((self.computer.weights - mean) / std)
            self.line2.set_xdata([i for i in range(len(self.computer.weights))])
            self.figCanvas2.draw()

    def addsuccess(self,t):
        #print(t)
        X = self.line3.get_xdata().copy()
        Y = self.line3.get_ydata().copy()
        #print(X,Y)
        Y = np.append(Y,t)
        X = np.append(X,X[-1]+1) if not len(X) == 0 else np.append(X,0)
        #print(X,Y)
        self.ax3.set_ylim(0,1.1*max(Y))
        xmax = self.ax3.get_xlim()[1]
        if max(X)> xmax: self.ax3.set_xlim(0,1.8*xmax)

        mean,std = [np.mean(Y), np.std(Y)]
        #std = np.std(Y)
        title = 'time mean='+str(mean)[:7]+'; std='+str(std)[:7]
        self.ax3.set_title(title)
        self.line3.set_xdata(X)
        self.line3.set_ydata(Y)

    def drawsuccess(self):
        self.figCanvas3.draw()

    def movecar(self,xv):
        carcoords = self.myCanvas.coords(self.car)
        carcoords[0]+=10
        carcoords[1]+=10
        self.myCanvas.move(self.car,conf.scale*xv[0]-carcoords[0],-0.5*conf.scale*math.sin(3*xv[0])-carcoords[1])

    def clearCanvas(self):
        self.myCanvas.delete("all")





if __name__ == "__main__":

    def refresh():
        #renderer.drawtrajectory()
        #root.after(10,refresh)
        try:
            if not args.fast: renderer.drawtrajectory()
            renderer.drawweights()
            renderer.drawsuccess()
            root.after(10,refresh)
        except ValueError:
            #print('refresh warning')
            root.after(10,refresh)

    def close_window():
        stopevent.set()
        root.quit()


    parser = argparse.ArgumentParser(prog='mountain_car',description='reinforce-learn the mountain car')

    parser.add_argument('-p','--path',type=str,default=conf.defaultpath,help='specify path where load/save model data')
    parser.add_argument('-m','--mode',type=str,default='r',help='specify the mode; r - run a model or f - look for a model from scratch, c = continue looking for a model')
    parser.add_argument('-i','--xvinit',type=float, default=False, nargs=2,help='specify initial position ('+str(conf.xLOW)+','+str(conf.xHIGH)+') and velocity ('+str(conf.vLOW)+','+str(conf.vHIGH)+')')
    parser.add_argument('-n','--noruns',type=int,default=-1,help='specify number of runs; -1 means infinite')
    parser.add_argument('-r','--reset',default=False,const=True,action='store_const',help='reset when trajectory is stuck during learning')
    parser.add_argument('-t','--lag',type=float,default=0.001,help='time lag in the run mode')
    parser.add_argument('-f','--fast',default=False,const=True,action='store_const',help='fast mode disabling trajectory tracking')
    args = parser.parse_args()

    print('model path:',args.path)
    print('mode:',{'r':'run a model','f':'find a model from scratch','c':'continue looking for a model'}[args.mode])
    print('number of runs:',args.noruns)
    #print('initial position and velocity [x,v]:',{bool:'random',list:'fixed'}[type(args.xvinit)])



    stopevent = threading.Event()

    #renderer
    root = tkinter.Tk()
    renderer = render(root)
    renderer.setConfigure(args)
    renderer.drawscene()




    computer = compute()
    computer.setConfigure(args)
    computer.setModel('sarsa')
    computer.setRenderer(renderer)
    renderer.setComputer(computer)
    computer.setStop(stopevent)
    computer.setDaemon(True)
    computer.start()


    refresh()
    root.protocol("WM_DELETE_WINDOW",close_window)
    root.mainloop()
    close_window()
    #print('mainloop end')
