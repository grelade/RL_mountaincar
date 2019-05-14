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

    def loadQhat(self,file):
        with open(file, 'rb') as f:
            data = pickle.loads(f.read())
            self.policy = data['policy']
            self.weights = data['weights']
            f.close()

    def saveQhat(self,file):
        with open(file, 'wb') as f:
            pickle.dump({'policy':self.policy,'weights':self.weights}, f, pickle.HIGHEST_PROTOCOL)
            f.close()

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

        for n in range(0,100):
            if (type(args.xvinit)==list): xv = [args.xvinit[0],args.xvinit[1]]
            else: xv = [np.random.uniform(conf.xLOW,conf.xHIGH),np.random.uniform(conf.vLOW,conf.vHIGH)]

            S = self.xv_to_S(xv)
            A = self.choosemove(self.S_to_cS(S))
            #print(self.policy)
            print(n,'; init [x,v]=',xv)

            while not xv[0] == conf.xHIGH:
                self.collecttrajectory(xv)
                time.sleep(0.01)
                #print(xv)
                if self.stopevent.isSet(): return None
                else: output = self.renderer.movecar(xv)
                R,xv_prime = env(xv,A)
                S_prime = self.xv_to_S(xv_prime)
                A_prime = self.choosemove(self.S_to_cS(S_prime))
                xv = xv_prime
                S = self.xv_to_S(xv)
                A = A_prime
            self.cleartrajectory()

    def sarsa(self):
        epsilon = conf.defaultepsilon
        alpha = 1
        gamma = conf.defaultgamma
        noepisodes = conf.defaultnoepisodes

        self.weights = [100 for i in range(0,len(self.cSA))]
        self.policy = self.greedy(epsilon)

        for n in range(0,noepisodes):
            xv = [np.random.uniform(conf.xLOW,conf.xHIGH),np.random.uniform(conf.vLOW,conf.vHIGH)]
            #xv = [0,0]
            S = self.xv_to_S(xv)
            A = self.choosemove(self.S_to_cS(S))
            self.policy = self.greedy(epsilon)

            print(n)
            visits = [1 for i in range(0,len(self.cSA))]
            propagation = 0
            while not xv[0] == conf.xHIGH:
                self.collecttrajectory(xv)
                #print(xv)
                #print(len(self.trajx),len(self.trajv))
                #if propagation<len(self.trajx): propagation=len(self.trajx)
                #else: input()
                #print(visits)
                #input()
                #visits[self.SA_to_iSA([S,A])]+=1
                if self.stopevent.isSet(): return None
                else: output = self.renderer.movecar(xv)
                R,xv_prime = env(xv,A)
                S_prime = self.xv_to_S(xv_prime)
                self.policy = self.greedy(epsilon)
                A_prime = self.choosemove(self.S_to_cS(S_prime))
                self.weights[self.SA_to_iSA([S,A])] += alpha/visits[self.SA_to_iSA([S,A])]*(R+ gamma*self.Qhat(self.SA_to_vSA([S_prime,A_prime])) - self.Qhat(self.SA_to_vSA([S,A])))
                xv = xv_prime
                S = self.xv_to_S(xv)
                A = A_prime
            #input()
            self.cleartrajectory()

    def run(self):
        if self.args.mode == 'r':
            self.loadQhat(self.args.path)
            self.simulate()
        elif self.args.mode == 'f':
            self.model()
            self.saveQhat(self.args.path)


class render():

    def __init__(self,root):
        self.root = root


        self.myCanvas = tkinter.Canvas(self.root, bg="white", height=conf.scaley, width=conf.scalex)
        self.myCanvas.xview_scroll(250,"pages")
        self.myCanvas.yview_scroll(250,"pages")
        self.myCanvas.pack(side='top')

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


        self.figCanvas = FigureCanvasTkAgg(self.fig,master=self.root)
        #self.myCanvas.show()

        self.figCanvas.get_tk_widget().pack(side='bottom')
        #self.frame.pack()

        self.buttonFrame = tkinter.Frame(self.root, bg="white")
        # self.buttonDraw = tkinter.Button(self.buttonFrame,text="drawtrajectory()", command=self.drawtrajectory)
        # self.buttonDraw.pack(side="left")
        self.buttonSave = tkinter.Button(self.buttonFrame,text="savePolicy()",command=self.savepolicy)
        self.buttonSave.pack(side="left")
        self.buttonFrame.pack()

    def setConfigure(self,args):
        self.args = args

    def setComputer(self,computer):
        self.computer = computer

    def savepolicy(self):
        self.computer.saveQhat(self.args.defaultpath)

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
        print(self.myCanvas.coords(self.car))

    def drawtrajectory(self):
        self.line.set_ydata(self.computer.trajv)
        self.line.set_xdata(self.computer.trajx)
        self.figCanvas.draw()

    def movecar(self,xv):
        carcoords = self.myCanvas.coords(self.car)

        self.myCanvas.move(self.car,conf.scale*xv[0]-carcoords[0],-0.5*conf.scale*math.sin(3*xv[0])-carcoords[1])

    def clearCanvas(self):
        self.myCanvas.delete("all")





if __name__ == "__main__":

    def refresh():
        try:
            renderer.drawtrajectory()
            root.after(10,refresh)
        except :
            print('refresh warning')
            root.after(10,refresh)

    def close_window():
        stopevent.set()
        root.destroy()

    parser = argparse.ArgumentParser(prog='mountain_car',description='reinforce-learn the mountain car')

    parser.add_argument('-p','--path',type=str,default=conf.defaultpath,help='specify path where load/save model data')
    parser.add_argument('-m','--mode',type=str,default='r',help='specify the mode; r - run a model or f - find a model')
    parser.add_argument('-i','--xvinit',type=float, default=False, nargs=2,help='specify initial position ('+str(conf.xLOW)+','+str(conf.xHIGH)+') and speed ('+str(conf.vLOW)+','+str(conf.vHIGH)+')')
    args = parser.parse_args()

    print('model path:',args.path)
    print('mode:',{'r':'run the model','f':'find the model'}[args.mode])
    print('initial values:',{bool:'random',list:'fixed'}[type(args.xvinit)])



    stopevent = threading.Event()

    #renderer
    root = tkinter.Tk()
    renderer = render(root)
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
