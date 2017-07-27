#! /usr/bin/env python
from math import floor, sqrt, pi
import numpy as np
from random import shuffle
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
from scipy import ndimage
import sys, os, re, string
from netCDF4 import Dataset
import time as time_lib
from scipy.sparse import lil_matrix, csc_matrix, hstack
import logging
import time

class Tools(object):


    def build_weight_array(self, array, fix_edges = False, normalize = False):
        '''
        Create np.array((8,L,W)) of quantity a in each of the neighbors to a cell
        '''

        a_shape = array.shape

        wgt_array = np.zeros((8, a_shape[0], a_shape[1]))
        nums = range(8)

        wgt_array[nums[0],:,:-1] = array[:,1:] # E
        wgt_array[nums[7],1:,:-1] = array[:-1,1:] # NE
        wgt_array[nums[6],1:,:] = array[:-1,:] # N
        wgt_array[nums[5],1:,1:] = array[:-1,:-1] # NW
        wgt_array[nums[4],:,1:] = array[:,:-1] # W
        wgt_array[nums[3],:-1,1:] = array[1:,:-1] # SW
        wgt_array[nums[2],:-1,:] = array[1:,:] # S
        wgt_array[nums[1],:-1,:-1] = array[1:,1:] # SE

        if fix_edges:
            wgt_array[nums[0],:,-1] = wgt_array[nums[0],:,-2]
            wgt_array[nums[7],:,-1] = wgt_array[nums[1],:,-2]
            wgt_array[nums[1],:,-1] = wgt_array[nums[7],:,-2]
            
            wgt_array[nums[7],0,:] = wgt_array[nums[1],1,:]
            wgt_array[nums[6],0,:] = wgt_array[nums[2],1,:]
            wgt_array[nums[5],0,:] = wgt_array[nums[3],1,:]
            wgt_array[nums[5],:,0] = wgt_array[nums[3],:,1]
            wgt_array[nums[4],:,0] = wgt_array[nums[4],:,1]
            wgt_array[nums[3],:,0] = wgt_array[nums[5],:,1]
            wgt_array[nums[3],-1,:] = wgt_array[nums[5],-2,:]
            wgt_array[nums[2],-1,:] = wgt_array[nums[6],-2,:]
            wgt_array[nums[1],-1,:] = wgt_array[nums[7],-2,:]

        if normalize:
            a_sum = np.sum(wgt_array, axis=0)
            wgt_array[:,a_sum!=0] = wgt_array[:,a_sum!=0] / a_sum[a_sum!=0]

        return wgt_array

    
    def direction_setup(self):
        '''set up grid with # of neighbors and directions'''
    
#         Nnbr = np.zeros((self.L,self.W), dtype=np.int)
        nbr = np.zeros((self.L,self.W,8))
        
        self.nbr = self.build_weight_array(np.ones((self.L,self.W)))
        self.Nnbr = np.sum(self.nbr, axis=0, dtype=np.int)
#         
#         
#         L = self.L
#         W = self.W
#         
#         ################
#         #center nodes
#         ################
#         Nnbr[1:L-1, 1:W-1] = 8
#         nbr[1:L-1, 1:W-1, :] = [(k+1) for k in range(8)]  
#         
#         
#         ################
#         # left side
#         ################
#         Nnbr[0, 1:W-1] = 5
#         
#         for k in range(5):
#             nbr[0, 1:W-1, k] = (6 + (k + 1)) % 8
#             
#         nbr[0, 1:W-1, 1] = 8 #replace zeros with 8   
#           
#           
#         ################
#         # upper side
#         ################
#         Nnbr[1:L-1, W-1] = 5
#         
#         for k in range(5):
#             nbr[1:L-1, W-1, k] = (4 + (k + 1)) % 8
#             
#         nbr[1:L-1, W-1, 3] = 8 #replace zeros with 8   
#            
#            
#         ################
#         # lower side
#         ################
#         Nnbr[1:L-1, 0] = 5
#         
#         for k in range(5):
#             nbr[1:L-1, 0, k] = (k + 1) % 8   
#             
#             
#         ####################
#         # lower-left corner
#         ####################
#         Nnbr[0,0] = 3
#         
#         for k in range(3):
#             nbr[0, 0, k] = (k + 1) % 8
#         
#         
#         ####################
#         # upper-left corner
#         ####################
#         Nnbr[0, W-1] = 3
#         
#         for k in range(3):
#             nbr[0, W-1, k] = (6 + (k + 1)) % 8
#             
#         nbr[0, W-1, 1] = 8 #replace zeros with 8
#         
#         
#         
#         self.Nnbr = Nnbr
#         self.nbr = nbr
    
    
        
        


    #############################################
    ############## initialization ###############
    #############################################

    def get_var_name(self, long_var_name): 
        return self._var_name_map[ long_var_name ]



    def import_file(self):

#         if self.verbose: self.logger.info('Reading input file: ' + self.input_file)

        self.input_file_vars = dict()
        numvars = 0

        o = open(self.input_file, mode = 'r')

        for line in o:
            line = re.sub('\s$','',line)
            line = re.sub('\A[: :]*','',line)
            ln = re.split('\s*[\:\=]\s*', line)

            if len(ln)>1:

                ln[0] = string.lower(ln[0])

                if ln[0] in self._input_var_names:

                    numvars += 1

                    var_type = self._var_type_map[ln[0]]

                    ln[1] = re.sub('[: :]+$','',ln[1])

                    if var_type == 'string':
                        self.input_file_vars[str(ln[0])] = str(ln[1])
                    if var_type == 'float':
                        self.input_file_vars[str(ln[0])] = float(ln[1])
                    if var_type == 'long':
                        self.input_file_vars[str(ln[0])] = int(ln[1])
                    if var_type == 'choice':

                        ln[1] = string.lower(ln[1])

                        if ln[1] == 'yes' or ln[1] == 'true':
                            self.input_file_vars[str(ln[0])] = True
                        elif ln[1] == 'no' or ln[1] == 'false':
                            self.input_file_vars[str(ln[0])] = False
                        else:
                            print "Alert! The option for the 'choice' type variable " \
                                  "in the input file '" + str(ln[0]) + "' is unrecognized. " \
                                  "Please use only Yes/No or True/False as values.\n"

                else:
                    print "Alert! The input file contains an unknown entry. The variable '" \
                          + str(ln[0]) + "' is not an input variable for this model. Check " \
                          " the spelling of the variable name and only use the symbols : and = " \
                            "in variable assignments.\n"

        o.close()
        
        for k,v in self.input_file_vars.items():
            setattr(self, self.get_var_name(k), v)
        
#         if self.verbose: self.logger.info('Finished reading ' + str(numvars) + ' variables from input file.')


    def init_logger(self):
    
        pass
 
        
    def set_defaults(self):
    
        for k,v in self._var_default_map.items(): setattr(self, self._var_name_map[k], v)



    def create_dicts(self):
                                   
        self._input_var_names = self._input_vars.keys()

        self._var_type_map = dict()
        self._var_name_map = dict()
        self._var_default_map = dict()

        for k in self._input_vars.keys():
            self._var_type_map[k] = self._input_vars[k]['type']
            self._var_name_map[k] = self._input_vars[k]['name']
            self._var_default_map[k] = self._input_vars[k]['default']



    def set_constants(self):

        self.g = 9.81   # (gravitation const.)
    
        # i for x direction, same as change in column
        # j for y direction, same as change in row
    

        SQ05 = sqrt(0.5)
        
        
        self.dxn_ivec = [1,SQ05,0,-SQ05,-1,-SQ05,0,SQ05] #E --> clockwise
        self.dxn_jvec = [0,SQ05,1,SQ05,0,-SQ05,-1,-SQ05] #E --> clockwise
        
        self.dxn_iwalk = [1,1,0,-1,-1,-1,0,1] #E --> clockwise
        self.dxn_jwalk = [0,1,1,1,0,-1,-1,-1] #E --> clockwise
        
        self.dxn_dist = \
        [sqrt(self.dxn_iwalk[i]**2 + self.dxn_jwalk[i]**2) for i in range(8)]
        
        self.dxn_iwalk_inlet = self.dxn_iwalk[0] #x comp of inlet flow direction
        self.dxn_jwalk_inlet = self.dxn_jwalk[0] #y comp of inlet flow direction
        
        self.direction_setup() 
    
    
 
      
        
    def create_other_variables(self):
    
        self.init_Np_water = self.Np_water
        self.init_Np_sed = self.Np_sed
    
        self.dx = float(self.dx)
    
        self.theta_sand = self.coeff_theta_sand * self.theta_water
        self.theta_mud = self.coeff_theta_mud * self.theta_water
    
        self.U_dep_mud = self.coeff_U_dep_mud * self.u0
        self.U_ero_sand = self.coeff_U_ero_sand * self.u0
        self.U_ero_mud = self.coeff_U_ero_mud * self.u0
    
        self.L0 = max(1, int(round(self.L0_meters / self.dx)))
        self.N0 = max(3, int(round(self.N0_meters / self.dx)))
    
        self.L = int(round(self.Length/self.dx))        # num cells in x
        self.W = int(round(self.Width/self.dx))         # num cells in y
        
        self.set_constants()

        self.u_max = 2.0 * self.u0              # maximum allowed flow velocity
    
        self.C0 = self.C0_percent * 1/100.      # sediment concentration

        # (m) critial depth to switch to "dry" node
        self.dry_depth = min(0.1, 0.1*self.h0)
        self.CTR = floor(self.W / 2.) - 1

        self.gamma = self.g * self.S0 * self.dx / (self.u0**2)

        self.V0 = self.h0 * (self.dx**2)    # (m^3) reference volume (volume to
                                            # fill cell to characteristic depth)

        self.Qw0 = self.u0 * self.h0 * self.N0 * self.dx    # const discharge
                                                            # at inlet        
                                                                                                   
        self.qw0 = self.u0 * self.h0                # water unit input discharge
        self.Qp_water = self.Qw0 / self.Np_water    # volume each water parcel

        self.qs0 = self.qw0 * self.C0               # sed unit discharge

        self.dVs = 0.1 * self.N0**2 * self.V0       # total amount of sed added 
                                                    # to domain per timestep

        self.Qs0 = self.Qw0 * self.C0           # sediment total input discharge
        self.Vp_sed = self.dVs / self.Np_sed    # volume of each sediment parcel
    
        self.itmax = 2 * (self.L + self.W)      # max number of jumps for parcel
        self.size_indices = int(self.itmax/2)   # initial width of self.indices
        
        self.dt = self.dVs / self.Qs0           # time step size

        self.omega_flow = 0.9
        self.omega_flow_iter = 2. / self.itermax
 
        # number of times to repeat topo diffusion
        self.N_crossdiff = int(round(self.dVs / self.V0))
 
    
        # self.prefix
        self.prefix = self.out_dir
        
        if self.out_dir[-1] is not '/':
            self.prefix = self.out_dir + '/'
        
        self.label = ''
        
        if self.site_prefix:
            self.label += self.site_prefix + '_'
        if self.case_prefix:
            self.label += self.case_prefix + '_'

        if len(self.label) > 0:
            self.prefix += self.label
        else:
            self.label += self.prefix[:-1] + '_'


    def create_domain(self):
        '''
        Creates the model domain
        '''

        ##### empty arrays #####

        self.x, self.y = np.meshgrid(np.arange(0,self.W), np.arange(0,self.L))
    
        self.cell_type = np.zeros((self.L,self.W))
    
        self.eta = np.zeros((self.L,self.W)).astype(np.float32)
        self.stage = np.zeros((self.L,self.W))
        self.depth = np.zeros((self.L,self.W))

        self.qx = np.zeros((self.L,self.W))
        self.qy = np.zeros((self.L,self.W))
        self.qxn = np.zeros((self.L,self.W))
        self.qyn = np.zeros((self.L,self.W))
        self.qwn = np.zeros((self.L,self.W))
        self.ux = np.zeros((self.L,self.W))
        self.uy = np.zeros((self.L,self.W))
        self.uw = np.zeros((self.L,self.W))
        
        self.wet_flag, self.sfc_visit, self.sfc_sum, self.prepath_flag, self.boundflag, self.wall_flag = np.zeros((6,self.L,self.W))
        
        self.iseq = np.zeros((self.itmax,1))
        self.jseq = np.zeros((self.itmax,1))
        
    
        self.wgt_flat = np.zeros((self.L*self.W, 8))

        self.qs = np.zeros((self.L,self.W))
        self.Vp_dep_sand = np.zeros((self.L,self.W))
        self.Vp_dep_mud = np.zeros((self.L,self.W))





        ##### domain #####
        cell_land = 2
        cell_channel = 1
        cell_ocean = 0
        cell_edge = -1
        

        self.cell_type[:self.L0,:] = cell_land
        
        channel_inds = int(self.CTR - round(self.N0 / 2))
        self.cell_type[:self.L0, channel_inds:channel_inds + self.N0] = cell_channel
        
        self.wall_flag[self.cell_type == cell_land] = 1

        self.cell_type[self.cell_type == cell_land] = -2
        # reset the land cell_type to -2


        result = [(sqrt((i-3)**2 + (j-self.CTR)**2))
            for i in range(self.L)
            for j in range(self.W)]
        
        result =  np.reshape(result,(self.L,self.W))
        
        self.boundflag[result >= min(self.L-5, self.W/2-5)] = 1
        
    
        
        
        
        
        self.stage[:] = np.maximum(0, self.L0 - self.y - 1) * self.dx * self.S0
        self.stage[self.cell_type == cell_ocean] = 0.
        
        self.depth[self.cell_type == cell_ocean] = self.h0
        self.depth[self.cell_type == cell_channel] = self.h0

        self.qx[self.cell_type == cell_channel] = self.qw0
        self.qx[self.cell_type == cell_ocean] = self.qw0 / 5.
        self.qw = (self.qx**2 + self.qy**2)**(0.5)

        self.ux[self.depth>0] = self.qx[self.depth>0] / self.depth[self.depth>0]
        self.uy[self.depth>0] = self.qy[self.depth>0] / self.depth[self.depth>0]
        self.uw[self.depth>0] = self.qw[self.depth>0] / self.depth[self.depth>0]
        
    
    
    
        self.inlet = list(np.unique(np.where(self.cell_type == 1)[1]))
        self.eta[:] = self.stage - self.depth
        
        self.clim_eta = (-self.h0 - 1, 0.05)
        
        epsilon = 0.000001
        
        self.eta[:] = self.eta + np.random.rand(self.L,self.W) * epsilon
        self.stage[:] = self.stage + np.random.rand(self.L,self.W) * epsilon
        self.depth[:] = self.depth + np.random.rand(self.L,self.W) * epsilon
        
        
        
        
  
    def run_one_timestep(self):
        '''
        Run the time loop once
        '''
        
        timestep = self._time

        if self.verbose:
            print '-'*20
            print 'Time = ' + str(self._time)


        for iteration in range(self.itermax):
        
            self.water_route(timestep)
            

    def random_pick_inlet(self, choices, probs = None):
        '''
        Randomly pick a number from array choices weighted by array probs
        Values in choices are column indices

        Return a tuple of the randomly picked index for row 0
        '''

        if not probs:
            probs = np.ones((len(choices),))

        idx = np.random.choice(choices, p = probs / sum(probs))

        return idx
            
            
            
    def water_route(self, timestep):
        '''route all water parcels'''
        
        for itr in range(1, self.itermax+1):
        
            
            self.qxn[:] = 0; self.qyn[:] = 0; self.qwn[:] = 0
            
            self.wet_flag[self.depth >= self.dry_depth] = 1
            
            self.sfc_visit = 0 * self.sfc_visit #surface visit
            self.sfc_sum = 0 * self.sfc_sum #surface sum
            
            
            for n in range(self.Np_water):
            
                it = self.route_parcel()
                self.free_surf(it)
                
            print('np = %d' %n)
            
#             self.update_water(timestep, itr)
        
        
       
        
    def route_parcel(self):
        '''routes one parcel'''
        
        self.prepath_flag = 0 * self.prepath_flag
        #pre=path refers to one parcel
        
        
        self.iseq = 0 * self.iseq
        self.jseq = 0 * self.jseq
        
        
        self.water_continue = 1
        self.sfccredit = 1
        
        
        #####################################################################
        #first movement is in direction of flow in inlet
        ##################################################################### 

        self.px = 0
        self.py = self.random_pick_inlet(self.inlet) #choose channel cell to start from
        
        self.qxn[self.px,self.py] = (self.qxn[self.px,self.py] +
                                     self.dxn_iwalk_inlet)
        
        
        self.qyn[self.px,self.py] = (self.qyn[self.px,self.py] +
                                     self.dxn_jwalk_inlet)
        
        
        self.qwn[self.px,self.py] = (self.qwn[self.px,self.py] +
                                     self.Qp_water / self.dx / 2)
                                     
                                     
        
        #####################################################################
        # parcel tracking
        #####################################################################                             
        it = 0
        
        self.iseq[it] = self.px #initialize tracking parcel path
        self.jseq[it] = self.py
        
        
        while self.water_continue == 1 and it < self.itmax:
            #keep routing parcel as long as is valid
            
            self.prepath_flag[self.px,self.py] = 1
            #set marker that cell has been visited
            
            it += 1
            nk,weight = self.calc_weights(self.px, self.py) #nk, weight
            
            outputs = self.choose_path(it, nk, weight)
            #pxn, pyn, dist, istep, jstep
            
            
        pxn,pyn,dist,istep,jstep = outputs

        
        
        if dist > 0:
        
            self.qxn[self.px,self.py] = self.qxn[self.px,self.py] + istep / dist
            self.qyn[self.px,self.py] = self.qyn[self.px,self.py] + jstep / dist
            
            self.qwn[pxn,pyn] = (self.qwn[pxn,pyn] +
                                 self.Qp_water / self.dx / 2)
            
        return it
        
        
        
        
        
            
    def free_surf(self, it):
        '''calculate free surface after routing one water parcel'''
    
        itback = it #start with last cell the parcel visited
        
        Hnew = np.zeros((self.L,self.W))
        
        if ((self.boundflag[int(self.iseq[itback]),int(self.jseq[itback])])
            and (self.sfccredit == 1)):
            
            Hnew[int(self.iseq[itback]),int(self.jseq[itback])] = self.H_SL
            #if cell is in ocean, H = H_SL (downstream boundary condition)
            
            
            it0 = 0
            
            for it in xrange(itback-1, -1, -1):
                #counting back from last cell visited
                
                i = int(self.iseq[it])
                ip = int(self.iseq[it+1])
                j = int(self.jseq[it])
                jp = int(self.jseq[it+1])
                dist = sqrt((ip-i)**2 + (jp-j)**2)
                
                if dist > 0:
                
                    if it0 == 0:
                    
                        if ((self.uw[i,j] > self.u0 * 0.5) or
                            (self.depth[i,j] < 0.1*self.h0)):
                            #see if it is shoreline
                            
                            it0 = it
                            
                        dH = 0
                        
                    else:
                    
                        if self.uw[i,j] == 0:
                        
                            dH = 0
                            #if no velocity
                            #no change in water surface elevation
                            
                        else:
                        
                            dH = (self.S0 *
                                     (self.ux[i,j] * (ip - i) * self.dx +
                                      self.uy[i,j] * (jp - j) * self.dx) /
                                  self.uw[i,j])
                                  #difference between streamline and parcel path
                                  
                Hnew[i,j] = Hnew[ip,jp] + dH
                #previous cell's surface plus difference in H
                
                self.sfc_visit[i,j] = self.sfc_visit[i,j] + 1
                #add up # of cell visits
                
                self.sfc_sum[i,j] = self.sfc_sum[i,j] + Hnew[i,j]
                # sum of all water surface elevations
    
    
    
        
        
        
    def choose_py_start(self):
        '''choose inlet channel cell'''
        
        #rand_num = np.random.randint(0,len(self.py_start))
        
        
        return self.py_start[np.random.randint(0, len(self.py_start))]
    
    
    
    
    
    
    def get_dxn(self, i, j, k):
        '''get direction of neighbor i,j,k'''
        
        return int(self.nbr[i,j,k])



        
    def choose_prob(self, weight):
        '''choose next step
        based on weights of neighbors and a random number'''
        
        weight_n = weight / np.add.reduce(weight)
        weighted_dir = np.random.choice(range(8), p = weight_n)

        istep = self.dxn_iwalk[weighted_dir]
        jstep = self.dxn_jwalk[weighted_dir]
                
        return istep,jstep      
        
        
        

    
    
    
    
    
    
    def choose_random(self, px, py):
        '''choose next cell randomly'''
        
        pxn = px + np.random.randint(-1,2)
        #choose between -1, 0, 1 and add that much to current cell
        pxn = max(0,pxn) #x index can't go below 0
        
        pyn = py + np.random.randint(-1,2)
        
        
        return pxn,pyn






    def choose_path(self,it,nk,weight):
        '''choose next cell or do random walk, then route parcels'''
        
        if np.add.reduce(weight) > 0:
            #if weight is not all zeros, choose target cell by probability
            
            self.weight = weight
            
            step = self.choose_prob(weight)
            #istep, jstep
            
            istep = step[0]
            jstep = step[1]
            
            
            
        if np.add.reduce(weight) == 0:
            # if weight is all zeros, random walk
            
            pxn,pyn = self.choose_random(self.px, self.py)
            
            ntry = 0
            
            while self.wet_flag[pxn,pyn] == 0 and ntry < 5:
                # try 5 times to find random step that is valid path
                
                ntry = ntry + 1
                
                pxn, pyn = self.choose_random(self.px, self.py)
                
            istep = pxn - self.px
            jstep = pyn - self.py
            
            
            
        pxn = self.px + istep
        pyn = self.py + jstep
        
        dist = sqrt(istep**2 + jstep**2)
        
        
        if dist > 0:
        
            self.qxn[self.px,self.py] = self.qxn[self.px,self.py] + istep/dist
            self.qyn[self.px,self.py] = self.qyn[self.px,self.py] + jstep/dist
            
            self.qwn[self.px,self.py] = (self.qwn[self.px,self.py] +
                                         self.Qp_water / self.dx / 2)
            
            
            
            self.qxn[pxn,pyn] = self.qxn[pxn,pyn] + istep/dist
            self.qyn[pxn,pyn] = self.qyn[pxn,pyn] + jstep/dist
            
            self.qwn[pxn,pyn] = (self.qwn[pxn,pyn] + 
                                 self.Qp_water / self.dx / 2)
            
            
        self.px = pxn
        self.py = pyn
        
        self.iseq[it] = self.px
        self.jseq[it] = self.py
        
        
        #####################################################################
        #deal with loops
        #####################################################################
        
        if self.prepath_flag[self.px,self.py] == 1 and it > self.L0:
            #if cell has already been visited
            
            self.sfccredit = 0
            
            Fx = self.px - 1
            Fy = self.py - self.CTR
            
            Fw = sqrt(Fx**2 + Fy**2)
            
            if Fw != 0:
                self.px = self.px + round(Fx / Fw * 5)
                self.py = self.py + round(Fy / Fw * 5)
                
            self.px = max(self.px, self.L0)
            self.px = int(min(self.L - 2, self.px))
            
            self.py = max(1, self.py)
            self.py = int(min(self.W - 2, self.py))
            
        
        
        #####################################################################
        # check if it reached the edge
        #####################################################################
            
        if self.boundflag[self.px,self.py] == 1:
            #if reach boundary, stop
            
            self.water_continue = 0
            
            #### MP: is there no return if self.boundflag[] is not 1?
            #### MP: should add fail-safe of some sort
            
        return pxn, pyn, dist, istep, jstep
    

        
        
        
        
        
    def calc_weights(self, px, py):
        '''calculate routing weights and choosing path'''
        
        nk = int(self.Nnbr[px,py]) #number of neighbors
        
        weight, weight_int, weight_sfc = np.zeros((3,8,))
        
        dxn = np.where(self.nbr[:,px,py] == 1)[0]
        
        for k in range(nk):
        
            # steps and distance from neighbor
            pxn = px + self.dxn_iwalk[dxn[k]]
            pyn = py + self.dxn_jwalk[dxn[k]]
            dist = self.dxn_dist[dxn[k]]
            
       
            
            if self.wet_flag[pxn,pyn] == 1 and self.wall_flag[pxn,pyn] == 0:
            
                weight_sfc[dxn[k]] = (max(0,
                                     self.stage[px,py] - self.stage[pxn,pyn])
                                / dist)
                #weight based on inertia; previous downstream direction
                
                
                weight_int[dxn[k]] = (max(0,
                                     self.qx[px,py] * self.dxn_ivec[dxn[k]] +
                                     self.qy[px,py] * self.dxn_jvec[dxn[k]])
                                / dist)
                #weight based on gravity; water surface slope
                
                
        if np.add.reduce(weight_sfc) != 0:
        
            weight_sfc = weight_sfc / np.add.reduce(weight_sfc)
            #require that weight_sfc >= 0
            
            
            
        if np.add.reduce(weight_int) != 0:
        
            weight_int = weight_int / np.add.reduce(weight_int)
            #require that weight_int >= 0
            
            
            
            
        weight = self.gamma * weight_sfc + (1-self.gamma) * weight_int
        #this gives new routing direction
        
        for k in range(nk):
            
            # steps and distance from neighbor
            pxn = px + self.dxn_iwalk[dxn[k]]
            pyn = py + self.dxn_jwalk[dxn[k]]
            dist = self.dxn_dist[dxn[k]]
            
            
            if self.wet_flag[pxn,pyn] == 1:
            
                weight[dxn[k]] = self.depth[pxn,pyn]**self.theta_water * weight[dxn[k]]
                
                #### MP: why does it matter if weight is equal to zero?
#                 print weight[dxn[k]]
#                 
#                 assert weight[dxn[k]] > 0, 'Weight[k] < 0!'
                
                
                
        return nk, weight
       
        