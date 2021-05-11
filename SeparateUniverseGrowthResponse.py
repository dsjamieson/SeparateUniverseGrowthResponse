import numpy as np
from scipy.integrate import ode
from scipy.interpolate import interp1d
import warnings

class SeparateUniverseGrowthResponse(object) :
	'''
	Computes time evolutions of the separate universe growth response sourced by linear CDM+baryon perturbations
	precomputed in an instance of classy.Class. The growth response can be computed at each wave number tabulated
	in the transfer function interpolation tables accessed using Class.get_transfer.  
	'''
	def __init__(self, cls, quiet = False) :
		'''
		Initialization requires an instance of classy.Class that has 'output':'mTk, vTk' and 'z_pk' set early enough
		for the initial conditions chosen in the SeparateUniverseGrowthResponse.getGrowthResponse method. The instance of
		classy.Class must already be computed.
		'''
		self.cls = cls
		self.quiet = quiet
		if 'gauge' in cls.pars and ('new' in cls.pars['gauge'] or 'New' in cls.pars['gauge']) :
			raise Exception('Error: SeparateUniverseGrowthResponse instance of classy.Class must be computed in the synchronous gause')
		if  not 'vTk' in self.cls.pars['output'] or not ('mTk' in self.cls.pars['output'] or 'dTk' in self.cls.pars['output']) :
			raise Exception('Error: SeparateUniverseGrowthResponse instance of classy.Class \'output\' must contain \'mTk\' and \'vTk\'')
		if self.cls.pars['extra_metric_transfer_functions'] != 'yes' :
			raise Exception('Error: SeparateUniverseGrowthResponse instance of classy.Class must have \'extra_metric_transfer_functions\':\'yes\'')
		if 'z_pk' in self.cls.pars :
			self.z_max = np.max([float(tz) for tz in str( self.cls.pars['z_pk']).split(',')])
			if 'z_max_pk' in self.cls.pars :
				self.z_max = np.max([float(self.cls.pars['z_max_pk']), self.z_max])
		elif 'z_max_pk' in self.cls.pars :
				self.z_max = float(self.cls.pars['z_max_pk'])
		else :
			raise Exception('Error: SeparateUniverseGrowthResponse instance of classy.Class must have \'z_pk\' or \'z_max_pk\'')
		if self.cls.get_transfer() == {} :
			raise Exception('Error: SeparateUniverseGrowthResponse instance of classy.Class must be computed before initialization')		
		#
		# Background
		#
		bg_loga = -np.log(1. + self.cls.get_background()['z'])
		bg_loga[-1] = 0.
		self.fb = self.cls.Omega_b() / (self.cls.Omega_b() + self.cls.Omega0_cdm())
		self.fc = 1. - self.fb
		self.H = interp1d(bg_loga, self.cls.get_background()['H [1/Mpc]'])
		self.zeq = self.cls.z_eq()
		if self.zeq > self.z_max and not self.quiet:
			warnings.warn('SeparateUniverseGrowthResponse.__init__ instance of classy.Class does not compute perturbations early enough' +
						  'for radiation dominated era initial conditions')
		#
		# Separate universe growth response ODE coefficients
		#
		f = self.cls.get_background()['gr.fac. f']
		w = self.cls.get_background()['(.)p_tot'] / self.cls.get_background()['(.)rho_tot']
		Omega_cb = (self.cls.get_background()['(.)rho_cdm'] + self.cls.get_background()['(.)rho_b']) / self.cls.get_background()['(.)rho_crit']
		self.drag = interp1d(bg_loga, 0.5 * (1. - 3. * w + 4. * f))
		self.source1 = interp1d(bg_loga, 1.5 * Omega_cb)
		self.source2 = interp1d(bg_loga, 2. / 3. * f)
		#
		# Thermodynamics
		#
		th_loga = -np.log(1. + self.cls.get_thermodynamics()['z'])
		self.w_b = interp1d(th_loga, self.cls.get_thermodynamics()['w_b'])
		self.c2_b = interp1d(th_loga, self.cls.get_thermodynamics()['c_b^2'])
		
	def getSourceMode(self, loga, k_ind) :
		'''
		Interpolates the CDM+baryon perturbation and its first derivative with respect to log(a) for the mode 
		whose wave number is self.cls.get_transfer(z)['k (h/Mpc)'][k_ind] at log-scale factor log(a) = loga
		'''
		z = np.exp(-loga) - 1.
		try :
			H = self.H(loga)
			c2_b = self.c2_b(loga)
			w_b = self.w_b(loga)
			tf = self.cls.get_transfer(z)
		except ValueError as e :
			raise Exception("Error, redshift %.6e out of interpolation bounds", z)
		delta_cdm = -tf['d_cdm'][k_ind]
		d_delta_cdm_d_loga = 0.5 * tf['h_prime'][k_ind] * (1. + z) / self.H(loga)
		delta_b = -tf['d_b'][k_ind]
		d_delta_b_d_loga =  -3. * (c2_b - w_b) * delta_b  + (1. + w_b) * (tf['t_b'][k_ind] * (1. + z) / H + d_delta_cdm_d_loga)
		return self.fc * delta_cdm + self.fb * delta_b, self.fc * d_delta_cdm_d_loga + self.fb * d_delta_b_d_loga

	def getGrowthResponse(self, k_ind, logai = None, dloga = 1.e-2, nstep = 100, rtol = 1.e-3, atol = 1.e-10) :
		'''
		Numerically integrates the second order ODE for the time evolution of the separate universe
		growth response sourced by a linear CDM+baryon mode with wave length self.cls.get_transfer(z)['k (h/Mpc)'][k_ind].
		Initial conditions are set during radition domination, so loga should not be set too late or the numerical
		solution with have a transcient contribution. 
		'''
		if logai == None :
			logai = -np.log(self.z_max + 1.)
		elif self.zeq > 1. / np.exp(logai) - 1. and not self.quiet :
			 warnings.warn('SeparateUniverseGrowthResponse.getGrowthResponse initial time logai not early enough for radiation dominated era initial conditions')
		if self.z_max < 1. / np.exp(logai) - 1. :
			raise Exception('Error: SeparateUniverseGrowthResponse.getGrowthResponse initial time too early')
		delta_l, d_delta_l_d_loga = self.getSourceMode(logai, k_ind)
		fi = 3. / 2. * self.source2(logai)
		Ri = fi / (1. + fi) / 3. * delta_l
		dRi = 2. * Ri
		def getGrowthResponseDEQs(loga, state) :
			'''
			Second order ODE for separate universe growth response sourced by linear mode delta_l
			'''
			if loga > 0. :
				loga = 0.
			R, dR = state
			delta_l, d_delta_l_d_loga = self.getSourceMode(loga, k_ind)
			deqs = np.zeros(2)
			deqs[0]  = dR
			deqs[1]  = - self.drag(loga) * dR + self.source1(loga) * delta_l + self.source2(loga) * d_delta_l_d_loga
			return deqs
		#
		# Set up integrator and integrate over log(a)
		#
		growth_response_ode = ode(getGrowthResponseDEQs)
		growth_response_ode.set_integrator('lsoda', nsteps = nstep, rtol = rtol, atol = atol)
		growth_response_ode.set_initial_value([Ri, dRi], logai)
		logas = [logai]
		Rs = [Ri]
		stop_time = -2. * dloga
		while growth_response_ode.t < stop_time :
			growth_response_ode.integrate(growth_response_ode.t + dloga)
			if not growth_response_ode.successful() :
				if not self.quiet :
					warnings.warn("SeparateUniverseGrowthResponse.getGrowthResponse failed to integrate at log(a) = %.6e" % (growth_response_ode.t))
				break
			logas.append(growth_response_ode.t)
			Rs.append(growth_response_ode.y[0])
		growth_response_ode.integrate(0.)
		if growth_response_ode.successful() :
			logas.append(growth_response_ode.t)
			Rs.append(growth_response_ode.y[0])	   
		logas = np.array(logas)
		Rs = np.array(Rs)
		delta_ls = np.array([self.getSourceMode(loga, k_ind)[0] for loga in logas])
		return logas, Rs / delta_ls