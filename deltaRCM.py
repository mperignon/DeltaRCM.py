#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:15:38 2017

@author: rebecca
"""

#imports
from __future__ import division
import numpy as np
import math as m
import matplotlib.pylab as plt
import pdb
    
#commonly changed inputs        
f_bedload = 0.5 #% of sand
totaltimestep = 1000
L = 40  #domain size (# of cells in x-direction); typically on order of 100
W = 80 #domain size (# of cells in y-direction); W = 2L for semicircle growth
plotinterval = 10 
plotintervalstrat = 500 #record strata less frequently
runID = 7
itermax = 1 # # of interations of water routing
Np_water = 1000 # number of water parcels; typically 2000
Np_sed = 500 # number of sediment parcels

#operations used in the model        
class model_steps(object):
    
    def direction_setup(self): #set up grid with # of neighbors and directions
        Nnbr = np.zeros((L,W), dtype=np.int)
        nbr = np.zeros((L,W,8))
        #center nodes
        Nnbr[1:L-1,1:W-1] = 8
        nbr[1:L-1,1:W-1,:] = [(k+1) for k in range(8)]     
        # left side
        Nnbr[0,1:W-1] = 5
        for k in range(5):
            nbr[0,1:W-1,k] = (6+(k+1))%8
        nbr[0,1:W-1,1] = 8 #replace zeros with 8     
        # upper side
        Nnbr[1:L-1,W-1] = 5
        for k in range(5):
            nbr[1:L-1,W-1,k] = (4+(k+1))%8
        nbr[1:L-1,W-1,3] = 8 #replace zeros with 8      
        # lower side
        Nnbr[1:L-1,0] = 5
        for k in range(5):
            nbr[1:L-1,0,k] = (k+1)%8   
        # lower-left corner
        Nnbr[0,0] = 3
        for k in range(3):
            nbr[0,0,k] = (k+1)%8
        # upper-left corner
        Nnbr[0,W-1] = 3
        for k in range(3):
            nbr[0,W-1,k] = (6+(k+1))%8
        nbr[0,W-1,1] = 8 #replace zeros with 8
        self.Nnbr = Nnbr
        self.nbr = nbr
        
    def subsidence_setup(self): #set up subsidence
        self.sigma = np.zeros((L,W))
        sigma_max = 0.0*self.h0/1000
        sigma_min = -0.0*self.h0/1000
        sigma = [(j/W*(sigma_max-sigma_min)+sigma_min)
            for i in range(3,L)
            for j in range(W)]
        sigma =  np.reshape(sigma,(L-3,W))
        self.sigma[3:L,:] = sigma

    def setup(self): #define model parameters and set up grid
        self.CTR = int((W-1)/2) # center cell
        N0 = 5 # num of inlet cells
        self.omega_flow_iter = 2*1/itermax 
        strataBtm = 1 #bottom layer elevation
        self.dx = 50 #cell size, m
        self.u0 = 1.0 #(m/s) characteristic flow velocity/inlet channel velocity
        self.h0 = 5 # (m) characteristic flow depth/inlet channel depth; typically m to tens of m
        self.S0 = 0.0003*f_bedload+0.0001*(1-f_bedload) #characteristic topographic slope; typically 10^-4-10^-5
        V0 = self.h0*(self.dx**2) #(m^3) reference volume; volume to fill a channel cell to characteristic flow depth
        dVs = 0.1*N0**2*V0 #sediment volume added in each timestep; used to determine time step size;
        Qw0 = self.u0*self.h0*N0*self.dx
        C0 = 0.1*1/100 #sediment concentration
        Qs0 = Qw0*C0 #sediment total input discharge
        self.dt = dVs/Qs0 #time step size
        self.qw0 = Qw0/N0/self.dx #water unit input discharge
        self.qs0 = self.qw0*C0 #sediment unit input discharge
        self.Qp_water = Qw0/Np_water #volume of each water parcel
        self.Vp_sed = dVs/Np_sed #volume of each sediment parcel
        GRAVITY = 9.81
        self.u_max = 2.0*self.u0
        hB = 1.0*self.h0 #(m) basin depth
        self.H_SL = 0 # sea level elevation (downstream boundary condition)
        self.SLR = 0/1000*self.h0/self.dt #SLR per timestep
        self.dry_depth = min(0.1,0.1*self.h0) #(m) critical depth to switch to 'dry' node
        self.gamma = GRAVITY*self.S0*self.dx/self.u0/self.u0 #determines ratio of influence of inertia versus water surface gradient in calculating routing direction; controls how much water spreads laterally
        #parameters for random walk probability calc
        self.theta_water = 1.0 #depth dependence (power of h) in routing water parcels
        self.theta_sand = 2.0*self.theta_water # depth dependence (power of h) in routing sand parcels; sand in lower part of water column, more likely to follow topographic lows
        self.theta_mud = 1.0*self.theta_water # depth dependence (power of h) in routing mud parcels
        #sediment deposition/erosion related parameters
        self.beta = 3 #non-linear exponent of sediment flux to flow velocity
        self._lambda = 1.0 #"sedimentation lag" - 1.0 means no lag
        self.U_dep_mud = 0.3*self.u0 #if velocity is lower than this, mud is deposited
        self.U_ero_sand = 1.05*self.u0 #if velocity higher than this, sand eroded
        self.U_ero_mud = 1.5*self.u0 #if velocity higher than this, mud eroded
        #topo diffusion relation parameters
        self.alpha = 0.1 #0.05*(0.25*1*0.125) # topo-diffusion coefficient
        self.N_crossdiff = int(round(dVs/V0))
        #smoothing water surface
        self.Nsmooth = 10 #iteration of surface smoothing per timestep
        self.Csmooth = 0.9 #center-weighted surface smoothing
        #under-relaxation between iterations
        self.omega_sfc = 0.1 #under-relaxation coef for water surface
        self.omega_flow = 0.9 #under-relaxation coef for water flow
        #storage prep
        self.eta = np.zeros((L,W)) # bed elevation
        self.H = np.zeros((L,W)) #free surface elevation
        self.h = np.zeros((L,W)) #depth of water
        self.qx = np.zeros((L,W)) #unit discharge vector (x-comp)
        self.qy = np.zeros((L,W)) #unit discharge vector (y-comp)
        self.qw = np.zeros((L,W)) #unit discharge vector magnitude
        self.ux = np.zeros((L,W)) #velocity vector (x-comp)
        self.uy = np.zeros((L,W)) #velocity vector (y-comp)
        self.uw = np.zeros((L,W)) #velocity magnitude
        #value definition ##this could be a function?
        SQ05 = m.sqrt(0.5)
        SQ2 = m.sqrt(2)
        self.dxn_ivec = [1,SQ05,0,-SQ05,-1,-SQ05,0,SQ05] #E --> clockwise
        self.dxn_jvec = [0,SQ05,1,SQ05,0,-SQ05,-1,-SQ05] #E --> clockwise
        self.dxn_iwalk = [1,1,0,-1,-1,-1,0,1] #E --> clockwise
        self.dxn_jwalk = [0,1,1,1,0,-1,-1,-1] #E --> clockwise
        self.dxn_dist = [1,SQ2,1,SQ2,1,SQ2,1,SQ2] #E --> clockwise
        self.wall_flag = np.zeros((L,W))
        self.boundflag = np.zeros((L,W))
        result = [(m.sqrt((i-3)**2+(j-self.CTR)**2))
            for i in range(L)
            for j in range(W)]
        result =  np.reshape(result,(L,W))
        self.boundflag[result >= min(L-5,W/2-5)] = 1
        #initial setup
        self.L0 = 3
#        type_ocean = 0
        type_chn = 1
        type_sed = 2
        types = np.zeros((L,W))
        types[0:self.L0,:] = type_sed
        types[0:self.L0,int(self.CTR-round(N0/2)+1):int(self.CTR-round(N0/2)+N0+1)] = type_chn
        self.wall_flag[types>1] = 1 ##types function?
        #topo setup
        self.h[types==0] = hB
        self.h[types==1] = self.h0
        self.H[0,:] = max(0,self.L0-1)*self.dx*self.S0
        self.H[1,:] = max(0,self.L0-2)*self.dx*self.S0
        self.H[2,:] = max(0,self.L0-3)*self.dx*self.S0
        self.eta = self.H-self.h
        #flow setup; flow doesn't satisfy mass conservation
        self.qx[types==1] = self.qw0
        self.qx[types==0] = self.qw0/5
        self.qw = np.sqrt(self.qx**2+self.qy**2)
        self.ux[self.h>0] = self.qx[self.h>0]/self.h[self.h>0]
        self.uy[self.h>0] = self.qy[self.h>0]/self.h[self.h>0]
        self.uw[self.h>0] = self.qw[self.h>0]/self.h[self.h>0]
        self.direction_setup()
        self.subsidence_setup()
        self.wet_flag = np.zeros((L,W))
        self.py_start = np.arange(self.CTR-round(N0/2)+1,self.CTR-round(N0/2)+N0+1, dtype=np.int) #vector of inlet cells y coord
        self.px_start = 0
        self.dxn_iwalk_inlet = self.dxn_iwalk[0] #x comp of inlet flow direction
        self.dxn_jwalk_inlet = self.dxn_jwalk[0] #y comp of inlet flow direction
        self.itmax = 2*(L+W)
        #self.Hnew = np.zeros((L,W)) #temp water surface elevation before smoothing
        self.qxn = np.zeros((L,W)) #placeholder for qx during calculations
        self.qyn = np.zeros((L,W))
        self.qwn = np.zeros((L,W))
        self.sfc_visit = np.zeros((L,W)) # number of water parcels that have visited cell
        self.sfc_sum = np.zeros((L,W)) #sum of water surface elevations from parcels that have visited cell
        self.prepath_flag = np.zeros((L,W)) #flag for one parcel, to see if it should continue
        self.iseq = np.zeros((self.itmax,1)) #tracks which cells were visited by parcel
        self.jseq = np.zeros((self.itmax,1))
        self.qs = np.zeros((L,W))
        #prepare to record strata
        self.z0 = self.H_SL-self.h0*strataBtm #bottom layer elevation
        self.dz = 0.01*self.h0  #layer thickness
        zmax = int(round((self.H_SL+self.SLR*totaltimestep*self.dt+self.S0*L/2*self.dx-self.z0)/self.dz)) #max layer number
        strata0 =-1 #default value of none
        self.strata = np.ones((L,W,zmax))*strata0
        topz = np.zeros((L,W), dtype = np.int) #surface layer number
        topz = np.rint((self.eta-self.z0)/self.dz)
        topz[topz < 1] = 1
        topz[topz > zmax] = zmax
        self.zmax = zmax
        self.topz = topz
        self.strata_age = np.zeros((L,W))
        self.sand_frac = 0.5 + np.zeros((L,W))
        self.Vp_dep_sand = np.zeros((L,W))
        self.Vp_dep_mud = np.zeros((L,W))
        
    def choose_py_start(self): #choose inlet channel cell
        #rand_num = np.random.randint(0,len(self.py_start))
        return self.py_start[np.random.randint(0,len(self.py_start))]
    
    def get_dxn(self,i,j,k): #get direction of neighbor i,j,k
        return int(self.nbr[i,j,k])
    
    def choose_prob(self,weight,nk,sed,px,py): #choose next step based on weights of neighbors and a random number
        weight = weight/np.add.reduce(weight)
        weight_val = [np.add.reduce(weight[0:k+1]) for k in range(nk)]
        if sed > 0:
            step_rand = 1-np.random.random() #sed routing
        else:
            step_rand = np.random.random() #water routing
        dxn = [self.get_dxn(px,py,k) for k in range(nk)]
        for k in xrange(nk):
            #dxn = self.get_dxn(px,py,k)
            if step_rand < weight_val[k]: #move into first cell that's weight is more than random #
                istep = self.dxn_iwalk[dxn[k]-1]
                jstep = self.dxn_jwalk[dxn[k]-1]
                break
        return istep,jstep
    
    def choose_random(self,px,py): #choose next cell randomly
        pxn = px+np.random.randint(-1,2) #choose between -1, 0, 1 and add that much to current cell
        pyn = py+np.random.randint(-1,2)
        pxn = max(0,pxn) #x index can't go below 0
        return pxn,pyn

    def choose_path(self,it,nk,weight): #choose next cell or do random walk, then route parcels
        if np.add.reduce(weight) > 0: #if weight is not all zeros, choose target cell by probability
            step = self.choose_prob(weight,nk,0,self.px,self.py) #istep, jstep
            istep = step[0]
            jstep = step[1]
        if np.add.reduce(weight) == 0: # if weight is all zeros, random walk
            pxn,pyn = self.choose_random(self.px,self.py)
            ntry = 0
            while self.wet_flag[pxn,pyn] == 0 and ntry < 5:#try 5 times to find random step that is valid path
                ntry = ntry+1
                pxn, pyn = self.choose_random(self.px,self.py)
            istep = pxn-self.px
            jstep = pyn-self.py
        pxn = self.px+istep
        pyn = self.py+jstep
        dist = m.sqrt(istep**2+jstep**2)
        if dist > 0:
            self.qxn[self.px,self.py] = self.qxn[self.px,self.py]+istep/dist
            self.qyn[self.px,self.py] = self.qyn[self.px,self.py]+jstep/dist
            self.qwn[self.px,self.py] = self.qwn[self.px,self.py]+self.Qp_water/self.dx/2
            self.qxn[pxn,pyn] = self.qxn[pxn,pyn]+istep/dist
            self.qyn[pxn,pyn] = self.qyn[pxn,pyn]+jstep/dist
            self.qwn[pxn,pyn] = self.qwn[pxn,pyn]+self.Qp_water/self.dx/2
        self.px=pxn
        self.py=pyn
        self.iseq[it] = self.px
        self.jseq[it] = self.py
        #deal with loops
        if self.prepath_flag[self.px,self.py] == 1 and it > self.L0: #if cell has already been visited
            self.sfccredit = 0
            Fx = self.px-1
            Fy = self.py-self.CTR
            Fw = m.sqrt(Fx**2+Fy**2)
            if Fw != 0:
                self.px = self.px+round(Fx/Fw*5)
                self.py = self.py+round(Fy/Fw*5)
            self.px = max(self.px,self.L0)
            self.px = int(min(L-2,self.px))
            self.py = max(1,self.py)
            self.py = int(min(W-2,self.py))
        if self.boundflag[self.px,self.py] == 1: #if reach boundary, stop
            self.water_continue = 0
            return pxn,pyn,dist,istep,jstep
            
    def nbrs(self,dxn,px,py): #location and distance to neighbor in dxn
        pxn = px+self.dxn_iwalk[dxn-1]
        pyn = py+self.dxn_jwalk[dxn-1]
        dist = self.dxn_dist[dxn-1]
        return pxn,pyn,dist
        
    def walk(self,i,j,k):
        dxn = self.get_dxn(i,j,k)
        inbr = i+self.dxn_iwalk[dxn-1]
        jnbr = j+self.dxn_jwalk[dxn-1]
        return inbr,jnbr
        
    def calc_weights(self,px,py): #calculate routing weights and choosing path
        nk = self.Nnbr[px,py]#number of neighbors
        weight = np.zeros((8,1))
        weight_int = np.zeros((8,1)) 
        weight_sfc = np.zeros((8,1)) 
        dxn = [self.get_dxn(px,py,k) for k in range(nk)]
        for k in range(nk):
            pxn,pyn,dist = self.nbrs(dxn[k],px,py) #calculate weight of each neighbor by direction
            if self.wet_flag[pxn,pyn] == 1 and self.wall_flag[pxn,pyn] == 0:
                weight_sfc[k] = max(0,self.H[px,py]-self.H[pxn,pyn])/dist #weight based on inertia; previous downstream direction
                weight_int[k] = max(0,self.qx[px,py]*self.dxn_ivec[dxn[k]-1]+self.qy[px,py]*self.dxn_jvec[dxn[k]-1])/dist #weight based on gravity; water surface slope
        if np.add.reduce(weight_sfc) != 0:
            weight_sfc = weight_sfc/np.add.reduce(weight_sfc) #require that weight_sfc >= 0
        if np.add.reduce(weight_int) != 0:
            weight_int = weight_int/np.add.reduce(weight_int) #require that weight_int >= 0
        weight = self.gamma*weight_sfc + (1-self.gamma)*weight_int #this gives new routing direction
        for k in range(nk):
            pxn,pyn,dist = self.nbrs(dxn[k],px,py) #calculate weight of each neighbor by direction
            if self.wet_flag[pxn,pyn] == 1:
                weight[k] = self.h[pxn,pyn]**self.theta_water*weight[k]
            if weight[k] < 0:
                pdb.set_trace()
        return nk,weight
        
    def route_parcel(self): #routes one parcel
        self.prepath_flag = 0*self.prepath_flag #pre=path refers to one parcel
        self.iseq = 0*self.iseq
        self.jseq = 0*self.jseq
        self.water_continue = 1
        self.sfccredit = 1
        self.px = self.px_start
        self.py = self.choose_py_start() #choose channel cell to start from
        self.qxn[self.px,self.py] = self.qxn[self.px,self.py]+self.dxn_iwalk_inlet #first movement is in direction of flow in inlet
        self.qyn[self.px,self.py] = self.qyn[self.px,self.py]+self.dxn_jwalk_inlet
        self.qwn[self.px,self.py] = self.qwn[self.px,self.py]+self.Qp_water/self.dx/2
        it = 0
        self.iseq[it] = self.px #initialize tracking parcel path
        self.jseq[it] = self.py
        while self.water_continue == 1 and it < self.itmax: #keep routing parcel as long as is valid
            self.prepath_flag[self.px,self.py] = 1 #set marker that cell has been visited
            it += 1
            nk,weight = self.calc_weights(self.px,self.py) #nk, weight
            outputs = self.choose_path(it,nk,weight) #pxn, pyn, dist, istep, jstep
        pxn,pyn,dist,istep,jstep = outputs
        if dist > 0:
            self.qxn[self.px,self.py] = self.qxn[self.px,self.py]+istep/dist
            self.qyn[self.px,self.py] = self.qyn[self.px,self.py]+jstep/dist
            self.qwn[pxn,pyn] = self.qwn[pxn,pyn]+self.Qp_water/self.dx/2
        return it
            
    def free_surf(self,it): #calculate free surface after routing one water parcel
        itback = it #start with last cell the parcel visted
        Hnew = np.zeros((L,W))
        if self.boundflag[int(self.iseq[itback]),int(self.jseq[itback])] == 1 and self.sfccredit == 1:
            Hnew[int(self.iseq[itback]),int(self.jseq[itback])] = self.H_SL #if cell is in ocean, H = H_SL (downstream boundary condition)
            it0 = 0
            for it in xrange(itback-1,-1,-1): #counting back from last cell visited
                i = int(self.iseq[it])
                ip = int(self.iseq[it+1])
                j = int(self.jseq[it])
                jp = int(self.jseq[it+1])
                dist = m.sqrt((ip-i)**2+(jp-j)**2)
                if dist > 0:
                    if it0 == 0:
                        if self.uw[i,j] > self.u0*0.5 or self.h[i,j] < 0.1*self.h0: #see if it is shoreline
                            it0 = it
                        dH = 0
                    else:
                        if self.uw[i,j] == 0:
                            dH = 0 #if no velocity, no change in water surface elevation
                        else:
                            dH = self.S0*(self.ux[i,j]*(ip-i)*self.dx+self.uy[i,j]*(jp-j)*self.dx)/self.uw[i,j] #difference between streamline and parcel path
                Hnew[i,j] = Hnew[ip,jp]+dH #previous cell's surface plus difference in H
                self.sfc_visit[i,j] = self.sfc_visit[i,j]+1 #add up # of cell visits
                self.sfc_sum[i,j] = self.sfc_sum[i,j]+Hnew[i,j] #sum of all water surface elevations
    
    def flood_corr(self): #flooding dry wet correction
        for i in xrange(L):
            for j in range(W):
                if self.wet_flag[i,j] == 0: #locate dry nodes
                    for k in range(self.Nnbr[i,j]):
                        inbr,jnbr = self.walk(i,j,k)
                        if self.wet_flag[inbr,jnbr] == 1 and self.H[inbr,jnbr] > self.eta[i,j]: #make dry cells same water surface elevation as wet neighbor
                            self.H[i,j] = self.H[inbr,jnbr]            
    
    def update_water(self,timestep,itr): #update surface after routing all parcels ##could divide into 3 functions for cleanliness
        #update free surface
        Hnew = self.eta+self.h
        Hnew[Hnew < self.H_SL] = self.H_SL #water surface height not under sea level
        Hnew[self.sfc_visit > 0] = self.sfc_sum[self.sfc_visit > 0]/self.sfc_visit[self.sfc_visit > 0] #find average water surface elevation for a cell
        Htemp = Hnew #smooth newly calculated free surface
        for itsmooth in range(self.Nsmooth):
            Hsmth = Htemp
            for i in range(L):
                for j in range(W):
                    if self.boundflag[i,j] != 1: #locate non-boundary cells
                        sumH = 0
                        nbcount = 0
                        for k in range(self.Nnbr[i,j]): #for all neighbors of each cell
                            inbr,jnbr = self.walk(i,j,k)
                            if self.wall_flag[inbr,jnbr] == 0:
                                sumH = sumH+Hsmth[inbr,jnbr]
                                nbcount += 1
                        if nbcount != 0:
                            Htemp[i,j] = self.Csmooth*Hsmth[i,j]+(1-self.Csmooth)*sumH/nbcount #smooth if are not wall cells
        Hsmth = Htemp
        if timestep > 1:
            self.H = (1-self.omega_sfc)*self.H+self.omega_sfc*Hsmth #underrelaxation (damping for numerical stability)
        self.flood_corr()
        self.h = self.H-self.eta
        self.h[self.h<0] = 0
        #flow field and velocity  field
        #update flow field
        dloc = np.sqrt(self.qxn**2+self.qyn**2)
        self.qxn[dloc>0] = self.qwn[dloc>0]*self.qxn[dloc>0]/dloc[dloc>0]
        self.qyn[dloc>0] = self.qwn[dloc>0]*self.qyn[dloc>0]/dloc[dloc>0]
        if timestep > 1:
            if itr == 1:
                self.qx = self.qxn*self.omega_flow+self.qx*(1-self.omega_flow)
                self.qy = self.qyn*self.omega_flow+self.qy*(1-self.omega_flow)
            else:
                self.qx=self.qxn*self.omega_flow_iter+self.qx*(1-self.omega_flow_iter)
                self.qy=self.qyn*self.omega_flow_iter+self.qy*(1-self.omega_flow_iter)
        else:
            self.qx = self.qxn
            self.qy = self.qyn
        self.qw = np.sqrt(self.qx**2+self.qy**2)
        #apply upstream constant flux boundary condition
        self.qx[self.px_start,self.py_start] = self.qw0
        self.qy[self.px_start,self.py_start] = 0
        self.qw[self.px_start,self.py_start] = self.qw0
        #update velocity field
        self.uw[(self.h>self.dry_depth) & (self.qw>0)] = np.minimum(self.u_max,self.qw[(self.h>self.dry_depth) & (self.qw>0)]/self.h[(self.h>self.dry_depth) & (self.qw>0)])
        self.ux[(self.h>self.dry_depth) & (self.qw>0)] = self.uw[(self.h>self.dry_depth) & (self.qw>0)]*self.qx[(self.h>self.dry_depth) & (self.qw>0)]/self.qw[(self.h>self.dry_depth) & (self.qw>0)]
        self.uy[(self.h>self.dry_depth) & (self.qw>0)] = self.uw[(self.h>self.dry_depth) & (self.qw>0)]*self.qy[(self.h>self.dry_depth) & (self.qw>0)]/self.qw[(self.h>self.dry_depth) & (self.qw>0)]
        self.ux[(self.h<=self.dry_depth) | (self.qw<=0)] = 0 #set velocity of dry cells to zero
        self.uy[(self.h<=self.dry_depth) | (self.qw<=0)] = 0
        self.uw[(self.h<=self.dry_depth) | (self.qw<=0)] = 0   
        
    def water_route(self,timestep): #route all water parcels
        for itr in range(1,itermax+1):
            self.qxn = 0*self.qxn
            self.qyn = 0*self.qyn
            self.qwn = 0*self.qwn
            self.wet_flag = 0*self.wet_flag
            self.wet_flag[self.h >= self.dry_depth] = 1
            self.sfc_visit = 0*self.sfc_visit #surface visit
            self.sfc_sum = 0*self.sfc_sum #surface sum
            for n in range(1,Np_water+1):
                it = self.route_parcel()
                self.free_surf(it)
            print('np = %d' %n)
            self.update_water(timestep,itr)
        
    def update_u(self,px,py): #update velocities after erosion or depositon
        if self.qw[px,py] > 0:
            self.ux[px,py] = self.uw[px,py]*self.qx[px,py]/self.qw[px,py]
            self.uy[px,py] = self.uw[px,py]*self.qy[px,py]/self.qw[px,py]
        else:
            self.ux[px,py] = 0
            self.uy[px,py] = 0

    def deposit(self,Vp_dep,px,py): #deposit sand or mud
        eta_change_loc = Vp_dep/self.dx**2
        self.eta[px,py] = self.eta[px,py]+eta_change_loc #update bed
        self.h[px,py] = self.H[px,py]-self.eta[px,py]
        if self.h[px,py] < 0:
            self.h[px,py] = 0
        self.uw[px,py] = min(self.u_max,self.qw[px,py]/self.h[px,py])
        self.update_u(px,py)
        self.Vp_res = self.Vp_res - Vp_dep #update amount of sediment left in parcel
        
    def erode(self,Vp_ero,px,py): #erode sand or mud; total sediment mass is preserved but individual categories of sand and mud are not
        eta_change_loc = -Vp_ero/(self.dx**2)
        self.eta[px,py] = self.eta[px,py]+eta_change_loc
        self.h[px,py] = self.H[px,py]-self.eta[px,py]
        if self.h[px,py] < 0:
            self.h[px,py] = 0
        self.uw[px,py] = min(self.u_max,self.qw[px,py]/self.h[px,py])
        self.update_u(px,py)
        self.Vp_res = self.Vp_res+Vp_ero
        
    def sand_dep_ero(self,px,py): #decide if erode or deposit sand
        U_loc = self.uw[px,py]
        qs_cap = self.qs0*f_bedload/self.u0**self.beta*U_loc**self.beta ##
        qs_loc = self.qs[px,py]
        Vp_dep = 0
        Vp_ero = 0
        if qs_loc > qs_cap: #if more than transport capacity has gone through cell
            Vp_dep = min(self.Vp_res,(self.H[px,py]-self.eta[px,py])/4*(self.dx**2)) #deposit sand
            self.deposit(Vp_dep,px,py)
        elif U_loc > self.U_ero_sand and qs_loc < qs_cap: #erosion can only occur if haven't reached transport capacity
            Vp_ero = self.Vp_sed*(U_loc**self.beta-self.U_ero_sand**self.beta)/self.U_ero_sand**self.beta
            Vp_ero = min(Vp_ero,(self.H[px,py]-self.eta[px,py])/4*(self.dx**2)) 
            self.erode(Vp_ero,px,py)
        self.Vp_dep_sand[px,py] = self.Vp_dep_sand[px,py]+Vp_dep
            
    def mud_dep_ero(self,px,py): #decide if deposit or erode mud
        U_loc = self.uw[px,py]
        Vp_dep = 0
        Vp_ero = 0
        if U_loc < self.U_dep_mud:
            Vp_dep = self._lambda*self.Vp_res*(self.U_dep_mud**self.beta-U_loc**self.beta)/(self.U_dep_mud**self.beta)
            Vp_dep = min(Vp_dep,(self.H[px,py]-self.eta[px,py])/4*(self.dx**2)) #change limited to 1/4 local depth
            self.deposit(Vp_dep,px,py)
        if U_loc > self.U_ero_mud:
            Vp_ero = self.Vp_sed*(U_loc**self.beta-self.U_ero_mud**self.beta)/self.U_ero_mud**self.beta
            Vp_ero = min(Vp_ero,(self.H[px,py]-self.eta[px,py])/4*(self.dx**2)) #change limited to 1/4 local depth
            self.erode(Vp_ero,px,py)
        self.Vp_dep_mud[px,py] = self.Vp_dep_mud[px,py]+Vp_dep
            
    def sed_parcel(self,theta_sed,sed,px,py): #route one sediment parcel
        it = 0
        self.iseq[it] = px
        self.jseq[it] = py
        sed_continue = 1
        while sed_continue == 1 and it < self.itmax:
            #choose next with weights
            weight = np.zeros((8,1))
            it += 1
            nk = self.Nnbr[px,py]
            dxn = [self.get_dxn(px,py,k) for k in range(nk)]
            for k in range(nk):
                pxn,pyn,dist = self.nbrs(dxn[k],px,py) #pxn, pyn, dist
                weight[k] = (max(0,self.qx[px,py]*self.dxn_ivec[dxn[k]-1]+self.qy[px,py]*self.dxn_jvec[dxn[k]-1])**1.0*self.h[pxn,pyn]**theta_sed)/dist #qw instead of downstream direction for weights for sediment parcels
                if weight[k] < 0:
                    pdb.set_trace()
                if self.wet_flag[pxn,pyn] != 1:
                    weight[k] = 0 #doesn't allow dry nodes
                if self.wall_flag[pxn,pyn] != 0:
                    weight[k] = 0 #doesn't allow wall nodes
            if np.add.reduce(weight) == 0:
                for k in range(nk):
                    pxn,pyn,dist = self.nbrs(dxn[k],px,py) #pxn, pyn, dist
                    weight[k] = 1/dist
                    if self.wall_flag[pxn,pyn] == 1:
                        weight[k] = 0
            istep,jstep = self.choose_prob(weight,nk,sed,px,py) #choose a cell to move into #istep,jstep
            dist = m.sqrt(istep**2+jstep**2)
            #deposition and erosion
            if sed == 1: #sand
                if dist > 0:
                    self.qs[px,py] = self.qs[px,py]+self.Vp_res/2/self.dt/self.dx #exit accumulation
                px = px+istep
                py = py+jstep
                if dist > 0:
                    self.qs[px,py] = self.qs[px,py]+self.Vp_res/2/self.dt/self.dx #entry accumulation
                self.sand_dep_ero(px,py)
                if self.boundflag[px,py] == 1:
                    sed_continue = 0 
            if sed == 2: #mud
                px = px+istep
                py = py+jstep
                self.mud_dep_ero(px,py)
                if self.boundflag[px,py] == 1:
                    sed_continue = 0 

    def sand_route(self): #route sand parcels; topo diffusion
        theta_sed = self.theta_sand
        for np_sed in xrange(1,int(Np_sed*f_bedload)+1):
            self.Vp_res = self.Vp_sed
            self.itmax = 2*(L+W)
            px = self.px_start
            py = self.choose_py_start()
            self.qs[px,py] = self.qs[px,py]+self.Vp_res/2/self.dt/self.dx
            self.sed_parcel(theta_sed,1.px,py)
        print('np_sand = %d' %np_sed)
        #topo diffusion; introduces lateral erosion as sed can be moved from banks into channels ##should be a function, for cleanliness
        for crossdiff in range(self.N_crossdiff):
            eta_diff = self.eta
            for i in range(1,L-1):
                for j in range(1,W-1):
                    if self.boundflag[i,j] == 0 and self.wall_flag[i,j] == 0:
                        crossflux = 0
                        for k in range(self.Nnbr[i,j]):
                            inbr,jnbr = self.walk(i,j,k)
                            if self.wall_flag[inbr,jnbr] == 0:
                                crossflux_nb = self.dt/self.N_crossdiff*self.alpha*0.5*(self.qs[i,j]+self.qs[inbr,jnbr])*self.dx*(self.eta[inbr,jnbr]-self.eta[i,j])/self.dx #diffusion based on slope and sand flux
                                crossflux = crossflux+crossflux_nb
                                eta_diff[i,j] = eta_diff[i,j]+crossflux_nb/self.dx/self.dx
            self.eta = eta_diff
    
    def mud_route(self): #route mud parcels
        theta_sed = self.theta_mud
        for np_sed in xrange(1,int(Np_sed*(1-f_bedload))+1):
            self.Vp_res = self.Vp_sed
            px = self.px_start
            py = self.choose_py_start()
            self.sed_parcel(theta_sed,2,px,py)
        print('np_mud = %d' %np_sed)
    
    def sed_route(self): #route all sediment
        self.qs = 0*self.qs
        self.Vp_dep_sand = 0*self.Vp_dep_sand
        self.Vp_dep_mud = 0*self.Vp_dep_mud
        self.sand_route()
        self.mud_route()
    
    def update_sed(self,timestep): #updates after sediment routing; save stratigraphy
        self.sand_frac[self.Vp_dep_sand>0] = self.Vp_dep_sand[self.Vp_dep_sand>0]/(self.Vp_dep_mud[self.Vp_dep_sand>0]+self.Vp_dep_sand[self.Vp_dep_sand>0])
        self.sand_frac[self.Vp_dep_sand<0] = 0
        #save strata
        for px in xrange(L):
            for py in range(W):
                zn = round((self.eta[px,py]-self.z0)/self.dz) 
                zn = max(1,zn)
                if zn > self.zmax:
                    pdb.set_trace()
                if zn >= self.topz[px,py]:
                    for z in range(int(self.topz[px,py]),int(zn)+1): #+1 makes inclusive
                        self.strata[px,py,z-1] = self.sand_frac[px,py] 
                else:
                    for z in range(int(zn),int(self.topz[px,py])+1): #+1 makes inclusive
                        self.strata[px,py,z-1] = -1 
                    self.sand_frac[px,py] = self.strata[px,py,max(0,z-2)]#max(1,z-1)
                self.topz[px,py] = zn
        if timestep > 1000: #subsidence
            self.eta = self.eta - self.sigma
            self.h = self.H-self.eta
        self.flood_corr
        self.h = self.H - self.eta
        self.eta[self.px_start,self.py_start] = self.H[self.px_start,self.py_start] - self.h0 #upstream boundary condition - constant depth
        self.H_SL = self.H_SL+self.SLR*self.dt #sea level rise
        
    def save_plots(self,timestep): #feed in needed variables, save them
        np.save('%dH%d' %(runID,timestep), self.H)
        np.save('%deta%d' %(runID,timestep), self.eta)
        np.save('%dqw%d' %(runID,timestep), self.qw)
        fig = plt.figure()
        fig.set_size_inches(18.5,10.5)
        ax = fig.add_subplot(2,2,1)
        ax.set_aspect('equal')
        plt.imshow(self.eta,interpolation='nearest',cmap=plt.cm.gray,vmin=-5,vmax=1) ##check max
        plt.colorbar().ax.set_ylabel('elevation (m)')
        ax = fig.add_subplot(2,2,4)
        plt.imshow(self.H, interpolation='nearest',cmap=plt.cm.Greys, vmin=0,vmax=0.25) ##check these maxes
        plt.colorbar().ax.set_ylabel('water surface elevation')
        ax = fig.add_subplot(2,2,3)
        plt.imshow(self.qw, interpolation='nearest',cmap=plt.cm.Blues,vmin=0,vmax=10) ##check max
        plt.colorbar().ax.set_ylabel('discharge')
        fname = '%ddelta%d'%(runID,timestep)
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
        
    def save_strat(self,timestep):
        np.save('%dstrata%d' %(runID,timestep), self.strata)

    def run_timestep(self,timestep): #run one model timestep
        self.water_route(timestep)
        self.sed_route()
        self.update_sed(timestep)
           
class DeltaRCM(model_steps): #define the model itself 
    
    def run(self):
        for timestep in xrange(1,totaltimestep+1):
            print('t = %d' %timestep)
            self.run_timestep(timestep)
            if timestep%plotinterval == 0: 
                self.save_plots(timestep)
            if timestep%plotintervalstrat == 0:
                self.save_strat(timestep)
    
    def __init__(self):
        self.setup()   
      
delta = DeltaRCM()
delta.run() #run the model
