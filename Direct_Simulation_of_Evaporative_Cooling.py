import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True,
                     "figure.dpi": 100,
                     "font.size": 16
                    })
import tqdm

class evap_cool_sim:
    """
        Direct Simulation Monte-Carlo simulation of a classical non-interacting gas in a harmonic
        potential.
        
        ...
        
        Attributes
        ----------
            N : int
                Number of particles in system.
            T : float
                Temperatures of system in K.
            mass : float
                Mass of each particle in kg.
            omega_ho : float
                Geometric average angular frequency, omega_ho = (omega_x*omega_y*omega_z)^1/3
            omega : array
                1D array length 3 of dimentionless angular frequencies (omega_x,omega_y,omega_z)/omega_HO.
            k_B : float
                Internal value of Boltzman constant. Here 1.38064852e-23 J/K
            hbar : float
                Internal value of the reduced Planck's constant. Here 1.0545718e-34 Js
            r : array
                Array of shape (Particle Number, 3). The dimensionless x, y and z positions of the particles,
                in units of a_HO.
            v : array
                Array of shape (Particle Number, 3). The dimensionless v_x, v_y and v_z velocities of the particles,
                in units of a_HO*omega_HO.
            N_eff : int
                Number of effective atoms each simulation particle corresponds to.
            r_history : list
                List of r arrays at a series of chosen time steps during the simulation.
            v_history : list
                List of v arrays at a series of chosen time steps during the simulation.
            std_r_history : array
                Array of shape (iterations, 3) showing the std deviations of the particles x y and z coords at
                every time step.
            std_v_history : array
                Array of shape (iterations, 3) showing the std deviations of the particles vx vy and vz coords at
                every time step.
            N_history : array
                Array of number of particles at every time step.
            T_history : array
                Array of temperature (assuming distribution is Gaussian) at every time step.
            dt : float
                Time step per step in units of (1/omega_HO).
        
        Methods
        -------
            initial_distribution():
                Create a distribution of velocities and particles randomly distributed
                across real and velocity space.
    """
    def __init__(self, omega_x, omega_y, omega_z):
        """
            Parameters
            ----------
                omega_x : float
                    Angular frequency of trapping potential in x direction in Hz.
                omega_y : float
                    Angular frequency of trapping potential in y direction in Hz.
                omega_z : float
                    Angular frequency of trapping potential in z direction in Hz.
        """
        
        self.omega_ho = (omega_x*omega_y*omega_z)**(1/3)
        self.omega = np.array([omega_x,omega_y,omega_z])/((omega_x*omega_y*omega_z)**(1/3))
        self.steps = None
        self.dt = None
        self.k_B = 1.38064852e-23 # Boltzmann constant in J/K
        self.hbar = 1.0545718e-34 # Reduced Planck's constant in Js
        self.N_eff = 1
        
        self.r = np.empty(0)
        self.v = np.empty(0)
        
        self.r_history = []
        self.v_history = []
        self.std_r_history = np.empty((0,3))
        self.std_v_history = np.empty((0,3))
        self.N_history = np.empty(0)
        self.T_history = np.empty(0)
        self.t_history = np.empty(0)
        
        self.rng = np.random.default_rng()
        
    def initial_distribution(self,initial_temp,initial_particles,PDF="normal",width=None):
        """
            Create a set of particles randomly distributed across real and velocity
            space according to specified probability distributions.
            
            Parameters
            ----------
                initial_temp : float
                    The initial temperature in K.
                initial_particles : int
                    Number of simulation particles to create.
                PDF : str
                    Either "flat" or "normal", for a flat distribution of positions and
                    velocities or a normal distribution of position and Maxwell-Boltzmann
                    distribution of velocity respectively.
                width : float
                    Width of the flat distribution in a_HO. Else scaled by (k_B*T)/(hbar*omega_ho).
        """
        
        if PDF == "normal":
            
            cov_r = np.diag(self.k_B*initial_temp/(self.omega_ho*self.hbar*(self.omega)**2))
            cov_v = np.diag(np.ones(3)*self.k_B*initial_temp/(self.hbar*self.omega_ho))
            
            self.r = np.random.multivariate_normal(mean=np.zeros(3),cov=cov_r,size=initial_particles)
            self.v = np.random.multivariate_normal(mean=np.zeros(3),cov=cov_v,size=initial_particles)
        
        if PDF == "flat":
            if width == None:
                self.r = (self.rng.random((initial_particles,3))*2 - 1)*self.k_B*initial_temp/(2*self.hbar*self.omega_ho)
                self.v = self.rng.random((initial_particles,3))*2 - 1
            else:
                self.r = (self.rng.random((initial_particles,3))*2 - 1)*width/2
                self.v = (self.rng.random((initial_particles,3))*2 - 1)*width/2
    
    def _get_collision_vel(self,v1,v2):
        """
            Get the post collision velocities.
            
            Parameters
            ----------
                v1 : array
                    The first particle's velocity array([vx,vy,vz]) in a_HO*omega_HO.
                v2 : array
                    The second particle's velocity array([vx,vy,vz]) in a_HO*omega_HO.
            Returns
            -------
                vfin: tuple of arrays
                    The two particle's velocities after a collision (v1fin,v2fin)
                    in a_HO*omega_HO.
        """

        q = 2*self.rng.random() - 1
        phi = 2*np.pi*self.rng.random()

        v_r = v1 - v2
        v_rmag = (np.sum(v_r**2)**0.5)
        v_cm = 0.5*(v1 + v2)

        v_rfin = v_rmag*np.array([((1-q**2)**0.5)*np.cos(phi),((1-q**2)**0.5)*np.sin(phi),q])

        return (v_cm + 0.5*v_rfin, v_cm - 0.5*v_rfin)
        
            
    def run_sim(self,steps,dt_secs,side_length,v_rmax,N_eff,radius,trap_depth,escape_prob="default",
                collisions=True,checkpoints="default"):
        """
            Run a Direct Simulation Monte-Carlo (DSMC) method simulation of hard sphere particles
            in a harmonic potential.
            
            Parameters
            ----------
                steps : int
                    The number of time steps to perform simulation over.
                dt_secs : float
                    The time step in seconds.
                side_length : float
                    The side length of a cube in a_HO for the DSMC method.
                v_rmax : float
                    The maximum relative velocity used by DSMC.
                radius : float
                    The radius of the particles in a_HO.
                trap_depth : func
                    Function trap_depth(t) of the trap depth with respect to time. t is array like
                    and in units of 1/omega_HO
                escape_prob : func
                    Function of potential and trap depth describing the probability of a particle
                    escaping the trap at that potential energy and trap depth.
                collisions : bool
                    True if collisions on, False if off.
                checkpoints: int
                    Number of times to save r and v.
        """
        self.N_eff = N_eff
        self.dt = dt_secs*self.omega_ho
        
        self.t_history = np.append(self.t_history,0)
        self._get_properties()

        for n in tqdm.tqdm(range(steps)):
            V = 0.5*np.sum((self.omega**2)*self.r**2,axis=1)
            
            if escape_prob=="default":
                probs = np.greater(V,trap_depth(n*self.dt))
            else:
                probs = escape_prob(V,trap_depth(n*self.dt))
            
            rand = self.rng.random(size=probs.shape)
            
            V_greater = np.greater(probs,rand)
            
            self.r = np.delete(self.r,V_greater,axis=0)
            self.v = np.delete(self.v,V_greater,axis=0)
            
            self.r = self.r + self.v*self.dt
            self.v = self.v -(self.omega**2)*self.r*self.dt
            
            
            if collisions == True:
                r_grid = self.r//side_length
                r_grid_max = np.amax(r_grid,axis=0)
                r_grid_min = np.amin(r_grid,axis=0)

                n_cells = np.prod((r_grid_max - r_grid_min)+1)

                n_warn = 0
                for i in np.arange(r_grid_min[0],r_grid_max[0]+1):
                    for j in np.arange(r_grid_min[1],r_grid_max[1]+1):
                        for k in np.arange(r_grid_min[2],r_grid_max[2]+1):
                            indices = np.nonzero((r_grid[:,0] == i)&(r_grid[:,1] == j)&(r_grid[:,2] == k))[0].tolist()
                            pairs = []
                            M_cand = int((1/(2*side_length**3))*(len(indices)**2)*np.pi*(radius**2)*v_rmax*self.N_eff*self.dt)
                            if len(indices) < 20:
                                n_warn += 1
                            if len(indices) > 1:
                                for m in range(M_cand):
                                    pairs.append(self.rng.choice(indices,size=(2),replace=False))
                                for pair in pairs:
                                    if (np.sum((self.v[pair[0]]-self.v[pair[1]])**2)**0.5)/v_rmax > self.rng.random():
                                        self.v[pair[0]],self.v[pair[1]] = self._get_collision_vel(self.v[pair[0]],self.v[pair[1]])
                                    

            if checkpoints == "default":
                self.t_history = np.append(self.t_history,(n+1)*self.dt)
                self._get_properties()
            elif (n+1)%(steps//checkpoints) == 0:
                self.t_history = np.append(self.t_history,(n+1)*self.dt)
                self._get_properties()
            else:
                self._get_properties(save_r_and_v=False)
    
    def _get_properties(self,save_r_and_v=True):
        """
            Get the thermodynamic properties of the gas.
            
            Parameters
            ----------
                save_r_and_v : bool
                    Boolean variable. If True the r and v coords will be saved to r and v history lists.
        """
        if save_r_and_v == True:
            self.r_history.append(self.r)
            self.v_history.append(self.v)
        
        std_r = np.std(self.r,axis=0)
        std_v = np.std(self.v,axis=0)
        self.std_r_history = np.vstack((self.std_r_history,std_r))
        self.std_v_history = np.vstack((self.std_v_history,std_v))
        
        self.N_history = np.append(self.N_history,len(self.r))
        self.T_history = np.append(self.T_history,((np.prod(std_v**2))**(1/3)*self.omega_ho*self.hbar/self.k_B))
        
    def plot_dist(self,step=-1,r_or_v="r",b=50):
        """
            Plot the x, y and z distributions of the simulation using Matplotlib.
            
            Parameters
            ----------
                step : int
                    The step of the simulation to plot. If run_sim set to "light" there will only be 10 steps
                r_or_v : string
                    If "r" position space plotted. If "v" velocity space plotted.
                b : int
                    2D and 1D histograms will have b*b and b bins respectively.
        """
        if r_or_v == "r":
            r = self.r_history[step]
            label = "$"
            units = "$"
        if r_or_v == "v":
            r = self.v_history[step]
            label = "$v_"
            units = "\\omega_{\\mbox{\\tiny $\\textrm{HO}$}}$"
        
        fig_dist, axs = plt.subplots(3,3,sharex=True,sharey="row",figsize=(10,10))
        axs[0,1].remove()
        axs[0,2].remove()
        axs[1,2].remove()
        
        axs[0,0].set_ylabel(label + "y / a_{\\mbox{\\tiny $\\textrm{HO}$}}"+units)
        axs[1,0].set_ylabel(label + "z / a_{\\mbox{\\tiny $\\textrm{HO}$}}"+units)
        axs[2,0].set_ylabel("$N$")
        
        axs[2,0].set_xlabel(label + "x / a_{\\mbox{\\tiny $\\textrm{HO}$}}"+units)
        axs[2,1].set_xlabel(label + "y / a_{\\mbox{\\tiny $\\textrm{HO}$}}"+units)
        axs[2,2].set_xlabel(label + "z / a_{\\mbox{\\tiny $\\textrm{HO}$}}"+units)
        
        axs[0,0].tick_params(bottom=False)
        axs[1,0].tick_params(bottom=False)
        axs[1,1].tick_params(left=False,bottom=False)
        axs[2,1].tick_params(left=False)
        axs[2,2].tick_params(left=False)
        
        axs[0,0].hist2d(r[:,0],r[:,1],bins=b,cmap=plt.cm.Greys)
        axs[1,0].hist2d(r[:,0],r[:,2],bins=b,cmap=plt.cm.Greys)
        axs[1,1].hist2d(r[:,1],r[:,2],bins=b,cmap=plt.cm.Greys)
    
        axs[2,0].hist(r[:,0],b,weights=np.ones(len(r))*self.N_eff,histtype='step')
        axs[2,1].hist(r[:,1],b,weights=np.ones(len(r))*self.N_eff,histtype='step')
        axs[2,2].hist(r[:,2],b,weights=np.ones(len(r))*self.N_eff,histtype='step')
        
        axs[0,0].set_aspect('equal')
        axs[1,0].set_aspect('equal')
        axs[1,1].set_aspect('equal')
            
        plt.tight_layout()
        plt.savefig("Corner Plot of Gas.pdf",bbox_inches='tight')
        plt.show()
        
    def plot_sigma(self):
        """
            Plot the change in sigma over time.
        """
        steps = len(self.std_r_history)
        
        
        fig_sigma, axs1 = plt.subplots(6,1,sharex=True,sharey=False,figsize=(10,10))
        axs1[5].set_xlabel("$t / \\omega_{\\mbox{\\tiny $\\textrm{HO}$}}^{-1}$")
        
        r_stds = self.std_r_history.T
        v_stds = self.std_v_history.T
        
        for i,y_label in enumerate(["$\\sigma_x / a_{\\mbox{\\tiny $\\textrm{HO}$}}$",
                                    "$\\sigma_y / a_{\\mbox{\\tiny $\\textrm{HO}$}}$",
                                    "$\\sigma_z / a_{\\mbox{\\tiny $\\textrm{HO}$}}$"]):
            axs1[i].plot(np.arange(steps)*self.dt,r_stds[i])
            axs1[i].set_ylabel(y_label)
        
        for i,y_label in enumerate(["$\\sigma_{vx} / a_{\\mbox{\\tiny $\\textrm{HO}$}}\\omega_{\\mbox{\\tiny $\\textrm{HO}$}}$",
                                    "$\\sigma_{vy} / a_{\\mbox{\\tiny $\\textrm{HO}$}}\\omega_{\\mbox{\\tiny $\\textrm{HO}$}}$",
                                    "$\\sigma_{vz} / a_{\\mbox{\\tiny $\\textrm{HO}$}}\\omega_{\\mbox{\\tiny $\\textrm{HO}$}}$"]):
            axs1[i+3].plot(np.arange(steps)*self.dt,v_stds[i])
            axs1[i+3].set_ylabel(y_label)
        plt.savefig("Standard Deviations.png",bbox_inches='tight')
        
    def save_run(self,run_name,file_path="./"):
        """
            Save a simulation run to a .npy binary file.
            
            Parameters
            ----------
                run_name : str
                    File name for the run.
                file_path : str
                    File path of where to save run. Default is current directory. 
        """
        params = [self.omega,self.omega_ho,self.dt,self.N_eff]
        
        data = [self.r_history,self.v_history,self.std_r_history,self.std_v_history,
                self.T_history,self.N_history,self.t_history,params]
        np.save(file_path+run_name+".npy",np.array(data,dtype=object),allow_pickle=True)
        
    def load_run(self,file_name):
        """
            Load a simulation run from a .npy binary file.
            
            Parameters
            ----------
                file_name : str
                    Name and path of .npy file to load.
        """
        
        data = np.load(file_name,allow_pickle=True)
        
        self.r_history, self.v_history, params = list(data[0]), list(data[1]), list(data[2])
        
        self.r_history = list(data[0])
        self.v_history = list(data[1])
        self.std_r_history = data[2]
        self.std_v_history = data[3]
        self.T_history = data[4]
        self.N_history = data[5]
        self.t_history = data[6]
        params = list(data[7])
        
        self.omega,self.omega_ho,self.dt,self.N_eff = params[0], params[1], params[2], params[3]

