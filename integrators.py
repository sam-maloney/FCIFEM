# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 09:20:59 2021

@author: Samuel A. Maloney

"""
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sp_la

from abc import ABCMeta, abstractmethod

class Integrator(metaclass=ABCMeta):
    """Class for time integration of the temporal ODE discretization.

    Attributes
    ----------
    LHS : scipy.sparse.csr_matrix
        LHS Matrix whose inverse action must be computed.
    RHS : scipy.sparse.csr_matrix
        RHS Matrix multiplying the current solution and added to forcing term.
    P : scipy.sparse.linalg.LinearOperator
        Preconditioner for the LHS matrix to be used with the linear solver.
    dt : float
        Time interval between each successive timestep.
    timestep : int
        Current timestep of the simulation.
    time : float
        Current time of the simulation; equal to timestep*dt.
    sim : FciFemSim
        Parent simulation class to which the integrator belongs.

    """

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    def __init__(self, fciFemSim, dt):
        """Initialize attributes of the time-integration scheme.

        Parameters
        ----------
        fciFemSim : FciFemSim
            Parent simulation class to which the integrator belongs.
        dt : float
            Time interval between each successive timestep.

        Returns
        -------
        None.

        """
        self.sim = fciFemSim
        self.time = 0.0
        self.timestep = 0
        self.dt = dt

    def precondition(self, P='ilu', **kwargs):
        """Generate and/or store the preconditioning matrix P.

        Parameters
        ----------
        P : {string, scipy.sparse.linalg.LinearOperator, None}, optional
            Which preconditioning method to use. P can be a LinearOperator to
            directly specifiy the preconditioner to be used. Otherwise it must
            be one of 'jacobi', 'ilu', or None. The default is 'ilu'.
        **kwargs
            Used to specify optional arguments for scipy.sparse.linalg.spilu.
            Only relevant if P is 'ilu', otherwise unsused.

        Returns
        -------
        None.

        """
        if isinstance(P, str):
            self.preconditioner = P.lower()
            if self.preconditioner == 'ilu':
                self.ilu = sp_la.spilu(self.LHS, **kwargs)
                self.P = sp_la.LinearOperator(self.LHS.shape,
                                              lambda x: self.ilu.solve(x))
            elif self.preconditioner == 'jacobi':
                self.P = sp_la.inv(sp.diags(self.LHS.diagonal(), format='csc'))
        elif P is None:
            self.P = None
            self.preconditioner = 'none'
        else:
            self.P = P
            self.preconditioner = 'UserDefined'

    def cond(self, order=2):
        """Compute the condition number of the LHS matrix of the solver.

        Parameters
        ----------
        order : {int, inf, -inf, ‘fro’}, optional
            Order of the norm. inf means numpy’s inf object. The default is 2.

        Returns
        -------
        c : float
            The condition number of the matrix.

        """
        if self.P != None:
            A = self.P @ self.LHS.A
        else:
            A = self.LHS
        if order == 2:
            LM = sp_la.svds(A, 1, which='LM', return_singular_vectors=False)
            SM = sp_la.svds(A, 1, which='SM', return_singular_vectors=False)
            c = LM[0]/SM[0]
        else:
            if sp.issparse(A):
                c = sp_la.norm(A, order) * sp_la.norm(sp_la.inv(A), order)
            else: # A is dense
                c = la.norm(A, order) * la.norm(la.inv(A), order)
        return c

    def step(self, nSteps = 1, **kwargs):
        """Integrate solution a given number of timesteps.

        Default implementation given for basic one-step schemes, but can be
        overriden for multi-step schemes as needed.

        Parameters
        ----------
        nSteps : int, optional
            Number of timesteps to compute. The default is 1.
        **kwargs
            Used to specify optional arguments passed to the linear solver.
            Note that kwargs["M"] will be overwritten, instead use
            precondition(...) to generate or specify a preconditioner.

        Returns
        -------
        None.

        """
        kwargs["M"] = self.P
        for i in range(nSteps):
            self.timestep += 1
            self.sim.u, info = sp_la.lgmres(self.LHS,
                self.RHS @ self.sim.u + self.sim.b, x0=self.sim.u, **kwargs)
            if (info != 0):
                print(f'TS {self.timestep}: solution failed with error '
                      f'code {info}')
        self.time = self.timestep * self.dt


class BackwardEuler(Integrator):
    @property
    def name(self): return 'BackwardEuler'

    def __init__(self, fciFemSim, R, M, dt, P='ilu', **kwargs):
        super().__init__(fciFemSim, dt)
        self.RHS = M / dt
        self.LHS = self.RHS - R
        self.precondition(P, **kwargs)


class CrankNicolson(Integrator):
    @property
    def name(self): return 'CrankNicolson'

    def __init__(self, fciFemSim, R, M, dt, P='ilu', **kwargs):
        super().__init__(fciFemSim, dt)
        self.RHS = M/dt + 0.5*R
        self.LHS = self.RHS - R
        self.precondition(P, **kwargs)


class LowStorageRK(Integrator):
    @property
    def name(self): return 'LowStorageRungeKutta'

    def __init__(self, fciFemSim, R, M, dt, P='ilu', betas=4, **kwargs):
        super().__init__(fciFemSim, dt)
        self.RHS = R
        self.dudt = np.zeros(self.sim.nNodes)
        if isinstance(betas, np.ndarray):
            self.betas = betas
        else:
            self.betas = np.array([1/n for n in range(betas, 0, -1)])
        if fciFemSim.massLumping:
            self.LHS = M.power(-1)
            self.step = self.stepMassLumped
        else:
            self.LHS = M
            self.step = self.stepNotMassLumped
            self.precondition(P, **kwargs)

    def stepNotMassLumped(self, nSteps = 1, **kwargs):
        kwargs["M"] = self.P
        for i in range(nSteps):
            uTemp = self.sim.u
            for beta in self.betas:
                self.dudt, info = sp_la.lgmres(self.LHS,
                    self.RHS @ uTemp + self.sim.b, x0=self.dudt, **kwargs)
                # self.dudt = sp_la.spsolve(self.LHS, self.RHS @ uTemp + self.sim.b)
                uTemp = self.sim.u + beta*self.dt*self.dudt
                if (info != 0):
                    print(f'TS {self.timestep}: solution failed with error '
                          f'code {info}')
            self.sim.u = uTemp
            self.timestep += 1
        self.time = self.timestep * self.dt

    def stepMassLumped(self, nSteps = 1, **kwargs):
        for i in range(nSteps):
            uTemp = self.sim.u
            for beta in self.betas:
                self.dudt = self.LHS @ (self.RHS @ uTemp + self.sim.b)
                uTemp = self.sim.u + beta*self.dt*self.dudt
            self.sim.u = uTemp
            self.timestep += 1
        self.time = self.timestep * self.dt
