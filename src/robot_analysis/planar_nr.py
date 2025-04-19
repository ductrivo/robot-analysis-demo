import matplotlib.animation as animation
import numpy as np
import sympy as sp
from IPython.display import HTML
from matplotlib import gridspec
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from sympy import (
    ImmutableDenseMatrix,
    Matrix,
    lambdify,
    simplify,
    symbols,
)

from robot_analysis.controller import FeedforwardPID

G = 9.81


class SimData:
    def __init__(self, n_links: int, n_max: int):
        self.t_full = np.zeros((1, n_max + 1))
        self.x_full = np.zeros((n_links * 2, n_max + 1))
        self.u_full = np.zeros((n_links, n_max + 1))
        self.tau_full = np.zeros((n_links, n_max + 1))
        self.pos_x_full = np.zeros((n_links, n_max + 1))
        self.pos_y_full = np.zeros((n_links, n_max + 1))


class PlanarRobotNR:
    def __init__(
        self,
        links: dict[str, list[float]],
        n_max: int = 500,
        t_step: float = 0.01,
    ) -> None:
        self._t_step = t_step
        self._n_max = n_max
        self.links = links
        self.tau_max = np.array(links['tau_max']).reshape((-1, 1))
        self._k: int = 0

        lengths = {len(v) for v in self.links.values()}
        if len(lengths) != 1:
            msg = 'Inconsistent lengths for link parameters.'
            raise ValueError(msg)

        self._n_links = len(self.links['l'])

        (
            self.eqs_sym,
            self.q_sym,
            self.dq_sym,
            self.ddq_sym,
            self.l_sym,
            self.m_sym,
            self.b_sym,
            self.L_sym,
            self.tau_sym,
            self.xC_sym,
            self.yC_sym,
        ) = self.build_model()

        self.M_sym, self.C_sym, self.G_sym = self.extract_M_C_G(
            self.eqs_sym,
            self.q_sym,
            self.dq_sym,
            self.ddq_sym,
        )

        self.mlb_sym = self.m_sym + self.l_sym + self.b_sym

        self.xC_func = lambdify(
            self.q_sym + self.mlb_sym,
            self.xC_sym,
            modules='numpy',
        )
        self.yC_func = lambdify(
            self.q_sym + self.mlb_sym,
            self.yC_sym,
            modules='numpy',
        )
        self.m_mat_func = lambdify(
            self.q_sym + self.mlb_sym,
            self.M_sym,
            modules='numpy',
        )
        self.c_mat_func = lambdify(
            self.dq_sym + self.q_sym + self.mlb_sym,
            self.C_sym,
            modules='numpy',
        )
        self.g_mat_func = lambdify(
            self.q_sym + self.mlb_sym,
            self.G_sym,
            modules='numpy',
        )
        self.tau_func = lambdify(
            self.ddq_sym + self.dq_sym + self.q_sym + self.mlb_sym,
            self.tau_sym,
            modules='numpy',
        )

        # Generate functions
        self.tau_func = self.create_tau_func(
            self.eqs_sym,
            self.q_sym,
            self.dq_sym,
            self.ddq_sym,
            self.mlb_sym,
        )

        self.data = SimData(n_links=self._n_links, n_max=self._n_max)

        print(
            f'Created robot with {self._n_links} link(s).'
            f'\nKinematics: End effector positions are:\n\txC={self.xC_sym}\n\ty_C={self.yC_sym}'
            f'\nDynamics: M, C, G matrices are:\n\tM={self.M_sym}\n\tC={self.C_sym}\n\tG={self.G_sym}'
        )

    @property
    def n_links(self) -> int:
        return self._n_links

    @property
    def t_step(self) -> float:
        return self._t_step

    @property
    def x(self) -> NDArray:
        return self.data.x_full[:, self._k, None]

    @x.setter
    def x(self, val: NDArray) -> None:
        self.data.x_full[:, self._k, None] = val

    @property
    def x_prev(self) -> NDArray:
        if self._k == 0:
            return np.zeros((self._n_links * 2, 1))
        return self.data.x_full[:, self._k - 1, None]

    @property
    def q(self) -> NDArray:
        return self.data.x_full[: self._n_links, self._k, None]

    @q.setter
    def q(self, val: NDArray) -> None:
        self.data.x_full[: self._n_links, self._k, None] = val

    @property
    def dq(self) -> NDArray:
        return self.data.x_full[self._n_links :, self._k, None]

    @dq.setter
    def dq(self, val: NDArray) -> None:
        self.data.x_full[self._n_links :, self._k, None] = val

    @property
    def dq_prev(self) -> NDArray:
        if self._k == 0:
            return np.zeros((self._n_links, 1))
        return self.data.x_full[self._n_links :, self._k - 1, None]

    @property
    def params(self):
        return (
            *self.q.flatten(),
            *self.links['l'],
            *self.links['m'],
            *self.links['b'],
        )

    @property
    def pos_x(self):
        return self.data.pos_x_full[:, self._k, None]

    @pos_x.setter
    def pos_x(self, val: NDArray):
        self.data.pos_x_full[:, self._k, None] = val

    @property
    def pos_y(self):
        return self.data.pos_y_full[:, self._k, None]

    @pos_y.setter
    def pos_y(self, val: NDArray):
        self.data.pos_y_full[:, self._k, None] = val

    @property
    def u(self) -> NDArray:
        return self.data.u_full[:, self._k, None]

    @u.setter
    def u(self, val: NDArray):
        self.data.u_full[:, self._k, None] = val

    @property
    def tau(self) -> NDArray:
        return self.data.tau_full[:, self._k, None]

    @tau.setter
    def tau(self, val: NDArray):
        self.data.tau_full[:, self._k, None] = val

    @property
    def m_mat(self):
        return self.m_mat_func(*self.params)

    @property
    def c_mat(self):
        return self.c_mat_func(
            *self.q.flatten(),
            *self.dq.flatten(),
            *self.links['m'],
            *self.links['l'],
            *self.links['b'],
        )

    @property
    def g_mat(self):
        return self.g_mat_func(*self.params)

    def simulate(self, controller: FeedforwardPID):
        """
        Simulate the dynamics over time using constant torque input.

        Parameters:
            x0 : np.ndarray
                Initial state vector [q1, ..., qn, dq1, ..., dqn]
            tau : np.ndarray
                Constant torque vector [tau1, ..., taun]
        """

        for k in range(1, self._n_max + 1):
            if controller.u_name == 'velocity':
                u = controller.make_step(y=self.q)
                dq_next = u
                q_next = self.q + self._t_step * dq_next
                ddq_next = (dq_next - self.dq) / self._t_step

                params = [
                    q_next.flatten(),
                    dq_next.flatten(),
                    self.links['m'],
                    self.links['l'],
                    self.links['b'],
                ]
                m_mat = self.compute_m_mat(*params)
                c_mat = self.compute_c_mat(*params)
                g_mat = self.compute_g_mat(*params)
                tau = m_mat @ ddq_next + c_mat + g_mat

            elif controller.u_name == 'torque':
                u = controller.make_step(y=self.q)
                tau = u

            else:
                tau = np.zeros((self._n_links, 1))

            # Need to check if the torque is sufficient
            tau = np.clip(tau, -self.tau_max, self.tau_max)

            ddq = np.linalg.pinv(self.m_mat) @ (tau - self.c_mat - self.g_mat)
            dq = self.dq_prev + self.t_step * ddq
            # print(f'np.vstack([dq, ddq]) = {np.vstack([dq, ddq])}')
            self._k = k
            self.data.t_full[0, self._k] = self._k * self._t_step
            self.u = u
            self.tau = tau
            self.x = self.x_prev + self._t_step * np.vstack([dq, ddq])
            self.pos_x = self.xC_func(*self.params)
            self.pos_y = self.yC_func(*self.params)

        print('Finished simulation, generating animation...')

    def compute_x_u_v(self, q, l, t):
        """
        Compute the symbolic center of mass positions and squared velocities
        for each link in a planar n-link manipulator.

        Parameters:
            q (list): Joint angle functions q1(t), ..., qn(t)
            l (list): Symbols l1, ..., ln for link lengths
            t (sympy.Symbol): Time symbol

        Returns:
            xC (list): x-coordinates of center of mass
            yC (list): y-coordinates of center of mass
            v_c_sq (list): squared velocities of center of mass
        """
        xC, yC, v_c_sq = [], [], []
        n = len(q)

        for i in range(n):
            angle_sum = sum(q[: i + 1])

            xi = sum([l[j] * sp.cos(sum(q[: j + 1])) for j in range(i)]) + (
                l[i] * sp.cos(angle_sum)
            )
            yi = sum([l[j] * sp.sin(sum(q[: j + 1])) for j in range(i)]) + (
                l[i] * sp.sin(angle_sum)
            )

            vxi = xi.diff(t)
            vyi = yi.diff(t)

            xC.append(xi)
            yC.append(yi)
            v_c_sq.append(vxi**2 + vyi**2)

        return xC, yC, v_c_sq

    def build_model(self):
        n = len(self.links['l'])
        t = sp.Symbol('t')

        # Generalized coordinates and their derivatives
        q = [sp.Function(f'q{i + 1}')(t) for i in range(n)]  # Joint angles
        dq = [qi.diff(t) for qi in q]  # Angular velocities
        ddq = [dqi.diff(t) for dqi in dq]  # Angular accelerations

        # Physical parameters: link lengths, masses, gravity
        l = sp.symbols(f'l1:{n + 1}')  # Lengths l1 to ln
        m = sp.symbols(f'm1:{n + 1}')  # Masses m1 to mn
        b = sp.symbols(f'b1:{n + 1}')  # Masses m1 to mn

        g = G  # sp.Symbol('g')  # Gravitational acceleration

        # Moments of inertia for rods: I_i = (1/12) * m_i * l_i^2
        inertia = [1 / 3 * m[i] * l[i] ** 2 for i in range(n)]

        # Center of mass positions and velocities
        xC, yC, v_c_sq = self.compute_x_u_v(q, l, t)

        # Kinetic energy (translational + rotational)
        T = sum(
            0.5
            * m[i]
            * v_c_sq[i]  # + 0.5 * inertia[i] * sum(dq[: i + 1]) ** 2
            for i in range(n)
        )

        # Potential energy
        V = sum(m[i] * g * yC[i] for i in range(n))

        # Lagrangian
        L = T - V
        L = sp.trigsimp(sp.simplify(L))

        # Generalized torques
        tau = [sp.Function(f'tau{i + 1}')(t) for i in range(n)]
        b = sp.symbols(f'b1:{n + 1}')  # viscous friction coefficients

        # Lagrange equations
        eqs = []
        for i in range(n):
            dL_dqi = L.diff(q[i])
            dL_ddqi = L.diff(dq[i])
            d_dt_dL_ddqi = dL_ddqi.diff(t)
            # eq = sp.Eq(sp.simplify(sp.trigsimp(d_dt_dL_ddqi - dL_dqi)), tau[i])
            # Add friction term to the equation: τ_i = d/dt(∂L/∂dq) - ∂L/∂q + b_i * dq_i
            eq = sp.Eq(
                sp.simplify(d_dt_dL_ddqi - dL_dqi + b[i] * dq[i]),
                tau[i],
            )

            eqs.append(eq)

        return (
            eqs,
            q,
            dq,
            ddq,
            list(l),
            list(m),
            list(b),
            L,
            tau,
            Matrix(xC),
            Matrix(yC),
        )

    def extract_M_C_G(self, eqs, q_sym, dq_sym, ddq_sym):
        """
        Extract symbolic matrices M(q), C(q, dq), G(q) from Lagrange equations,
        ensuring variables are treated as independent.

        Parameters:
            eqs: list of sympy.Eq
            ddq_sym: list of second derivatives (q̈i(t))
            dq_sym: list of first derivatives (qi̇(t))
            q_sym: list of positions (qi(t))

        Returns:
            M, C, G as ImmutableDenseMatrix
        """
        n = len(ddq_sym)

        # 1. Create independent symbols: q1, dq1, ddq1, ...
        q_flat = symbols(f'q1:{n + 1}')
        dq_flat = symbols(f'dq1:{n + 1}')
        ddq_flat = symbols(f'ddq1:{n + 1}')

        # 2. Substitution map: q1(t) -> q1, dq1(t) -> dq1, ...
        replace_dict = {}
        for i in range(n):
            replace_dict[q_sym[i]] = q_flat[i]
            replace_dict[dq_sym[i]] = dq_flat[i]
            replace_dict[ddq_sym[i]] = ddq_flat[i]

        M = Matrix.zeros(n)
        C = Matrix.zeros(n, 1)
        G = Matrix.zeros(n, 1)

        for i, eq in enumerate(eqs):
            expr = eq.lhs.subs(replace_dict)

            # 1. Mass matrix
            for j in range(n):
                M[i, j] = expr.diff(ddq_flat[j])

            # 2. Remove mass terms
            expr_minus_M = expr - sum(M[i, j] * ddq_flat[j] for j in range(n))

            # 3. Gravity vector: terms independent of dq
            G[i] = simplify(expr_minus_M.subs(dict.fromkeys(dq_flat, 0)))

            # 4. Coriolis matrix:
            C[i] = simplify(expr_minus_M - G[i])
            # for j in range(n):
            #     C[i, j] = C_expr.diff(dq_flat[j])

        reverse = {v: k for k, v in replace_dict.items()}

        M = M.subs(reverse)
        C = C.subs(reverse)
        G = G.subs(reverse)
        return (
            ImmutableDenseMatrix(M),
            ImmutableDenseMatrix(C),
            ImmutableDenseMatrix(G),
        )

    def create_tau_func(
        self,
        eqs,
        q_sym,
        dq_sym,
        ddq_sym,
        mlb_sym,
    ):
        """
        Generate a numerical function to compute joint torques tau from q, dq, ddq.

        Parameters:
            eqs: list of sympy Eq (from build_model)
            q_sym: list of sympy symbols/functions for q
            dq_sym: list of sympy expressions for dq
            ddq_sym: list of sympy expressions for ddq
            mlb_sym: list of additional symbols to include (e.g., l1, m1, g, ...)

        Returns:
            tau_func: a callable function tau_func(q_vals, dq_vals, ddq_vals, param_vals_dict)
        """
        n = len(q_sym)
        t = sp.Symbol('t')

        # Remove time dependency: q1(t) → q1, etc.
        q_flat = [sp.Symbol(f'q{i + 1}') for i in range(n)]
        dq_flat = [sp.Symbol(f'dq{i + 1}') for i in range(n)]
        ddq_flat = [sp.Symbol(f'ddq{i + 1}') for i in range(n)]

        replace_dict = {}
        for i in range(n):
            replace_dict[q_sym[i]] = q_flat[i]
            replace_dict[dq_sym[i]] = dq_flat[i]
            replace_dict[ddq_sym[i]] = ddq_flat[i]

        # Extract tau expressions from lhs of equations and substitute flat symbols
        tau_exprs = [eq.rhs.subs(replace_dict) for eq in eqs]

        # Create lambdified function
        tau_func_sympy = lambdify(
            q_flat + dq_flat + ddq_flat + mlb_sym,
            tau_exprs,
            modules='numpy',
        )

        def tau_func(q_vals, dq_vals, ddq_vals, mlb_sym):
            """
            Evaluate tau given q, dq, ddq and parameter values.

            Parameters:
                q_vals, dq_vals, ddq_vals: list or array of joint states
                param_vals_dict: dict of values for l1, m1, g, ...

            Returns:
                tau: np.array of joint torques
            """
            # Concatenate all inputs for lambdified function
            inputs = (
                list(q_vals) + list(dq_vals) + list(ddq_vals) + list(mlb_sym)
            )

            # Evaluate
            return np.array(tau_func_sympy(*inputs), dtype=np.float64)

        return tau_func

    def compute_m_mat(self, q, dq, m, l, b):
        return self.m_mat_func(*q, *m, *l, *b)

    def compute_c_mat(self, q, dq, m, l, b):
        return self.c_mat_func(*q, *dq, *m, *l, *b)

    def compute_g_mat(self, q, dq, m, l, b):
        return self.g_mat_func(*q, *m, *l, *b)

    def dynamics(self, q, dq, tau, m, l, b):
        """
        TODO: Consider to remove
        """
        M_val = self.m_mat_func(*q, *m, *l, *b).reshape((-1, 1))
        C_val = self.c_mat_func(*q, *dq, *m, *l, *b).reshape((-1, 1))
        G_val = self.g_mat_func(*q, *m, *l, *b).reshape((-1, 1))
        xC_val = self.xC_func(*q, *m, *l, *b)
        yC_val = self.yC_func(*q, *m, *l, *b)

        ddq = np.linalg.pinv(M_val) @ (tau - C_val - G_val)
        return np.concatenate([dq, ddq]), xC_val, yC_val


def animate_all(
    t_vals,
    xC_log,
    yC_log,
    x_log,
    u_log,
    tau_log,
    theta_d,
    tau_max,
    error,
    interval=5,
    save_path=None,
):
    T, n = xC_log.shape

    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(
        ncols=2,
        nrows=3,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.5,
    )

    # --- Subplot 1: Forward Kinematics
    ax0 = fig.add_subplot(spec[:, 0])
    ax0.set_xlim(-1.5, 1.5)
    ax0.set_ylim(-1.5, 1.5)
    ax0.set_aspect('equal')
    ax0.set_title(
        f'Trajectory\nτ_max={tau_max}, steady state error={error[0]:.1f}%'
    )
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid(True)

    (link_line,) = ax0.plot([], [], 'o-', color='black', lw=2)
    traj_lines = [ax0.plot([], [], '-', lw=1)[0] for _ in range(n)]
    traj_x, traj_y = [[] for _ in range(n)], [[] for _ in range(n)]

    # --- Subplot 2: Joint states x
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.set_title('Joint States')
    ax1.set_ylabel('theta')
    ax1.grid(True)
    ax1.axhline(y=theta_d, color='red', linestyle='--', label='Set point')

    n_joints = x_log.shape[1] // 2
    x_state_lines = [
        ax1.plot([], [], label=f'x{i + 1}')[0] for i in range(n_joints)
    ]
    ax1.legend()
    ax1.set_xlim(0, t_vals[-1])
    y_lim = theta_d * 1.5
    ax1.set_ylim(-y_lim, y_lim)

    # --- Subplot 3: Control
    ax2 = fig.add_subplot(spec[1, 1], sharex=ax1)
    ax2.set_title('Control input')
    ax2.set_ylabel('theta_dot')
    ax2.grid(True)
    u_state_lines = [
        ax2.plot([], [], label=f'u{i + 1}')[0] for i in range(u_log.shape[1])
    ]
    ax2.legend()
    ax2.set_xlim(0, t_vals[-1])
    u_lim = np.max(np.abs(u_log)) * 1.1
    ax2.set_ylim(-u_lim, u_lim)

    # --- Subplot 4: Torques τ
    ax3 = fig.add_subplot(spec[2, 1], sharex=ax1)
    ax3.set_title('Torques')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Torque [Nm]')
    ax3.grid(True)
    tau_lines = [
        ax3.plot([], [], label=f'τ{i + 1}')[0] for i in range(tau_log.shape[1])
    ]
    ax3.legend()
    ax3.set_xlim(0, t_vals[-1])
    tau_lim = np.max(np.abs(tau_log)) * 1.1
    ax3.set_ylim(-tau_lim, tau_lim)

    # History logs
    x_state_hist = [[] for _ in range(n_joints)]
    u_state_hist = [[] for _ in range(u_log.shape[1])]
    tau_hist = [[] for _ in range(tau_log.shape[1])]

    def init():
        link_line.set_data([], [])
        for line in traj_lines + x_state_lines + u_state_lines + tau_lines:
            line.set_data([], [])
        return [
            link_line,
            *traj_lines,
            *x_state_lines,
            *u_state_lines,
            *tau_lines,
        ]

    def update(frame):
        # Forward kinematics
        xs = [0, *list(xC_log[frame])]
        ys = [0, *list(yC_log[frame])]
        link_line.set_data(xs, ys)

        for i in range(n):
            traj_x[i].append(xC_log[frame, i])
            traj_y[i].append(yC_log[frame, i])
            traj_lines[i].set_data(traj_x[i], traj_y[i])

        for i in range(n_joints):
            x_state_hist[i].append(x_log[frame, i])
            x_state_lines[i].set_data(t_vals[: frame + 1], x_state_hist[i])

        for i in range(u_log.shape[1]):
            u_state_hist[i].append(u_log[frame, i])
            u_state_lines[i].set_data(t_vals[: frame + 1], u_state_hist[i])

        for i in range(tau_log.shape[1]):
            tau_hist[i].append(tau_log[frame, i])
            tau_lines[i].set_data(t_vals[: frame + 1], tau_hist[i])

        return [
            link_line,
            *traj_lines,
            *x_state_lines,
            *u_state_lines,
            *tau_lines,
        ]

    plt.close(fig)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        init_func=init,
        interval=interval,
        blit=True,
        repeat=False,
    )
    return HTML(ani.to_jshtml())


def plot_ax(robot):
    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(
        ncols=2,
        nrows=3,
        hspace=0.5,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
    )

    # --- Subplot 1: Forward Kinematics
    ax0 = fig.add_subplot(spec[:, 0])
    ax0.set_xlim(-2.5, 2.5)
    ax0.set_ylim(-2.5, 2.5)
    ax0.set_aspect('equal')
    # ax0.set_title(
    #     f'Trajectory\nτ_max={robot.tau_max}, steady state error={error[0]:.1f}%'
    # )
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid(True)

    # --- Subplot 2: Joint states x
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.set_title('Joint States')
    ax1.set_ylabel('theta')
    ax1.grid(True)
    # ax1.axhline(y=theta_d, color='red', linestyle='--', label='Set point')
    # ax1.legend()
    ax1.set_xlim(0, robot.data.t_full[0, -1])

    # --- Subplot 3: Control
    ax2 = fig.add_subplot(spec[1, 1], sharex=ax1)
    ax2.set_title('Control input')
    # ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('theta_dot')
    ax2.grid(True)
    ax2.set_xlim(0, robot.data.t_full[0, -1])
    # y_lim = np.max(np.abs(robot.data.u_full)) * 1.1
    # ax2.set_ylim(-y_lim, y_lim)

    # --- Subplot 4: Torques τ
    ax3 = fig.add_subplot(spec[2, 1], sharex=ax1)
    ax3.set_title('Torques')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Torque [Nm]')
    ax3.grid(True)
    ax3.set_xlim(0, robot.data.t_full[0, -1])
    y_lim = np.max(np.abs(robot.data.tau_full)) * 1.1
    y_min = np.min(robot.data.tau_full) * 1.1
    # ax3.set_ylim(-y_min, y_lim)

    return fig, ax0, ax1, ax2, ax3


def plot_final(
    robot: PlanarRobotNR,
    save_path=None,
):
    fig, ax0, ax1, ax2, ax3 = plot_ax(robot)

    # Plot robot links
    xs = [0, *list(robot.data.pos_x_full[:, -1])]
    ys = [0, *list(robot.data.pos_y_full[:, -1])]
    ax0.plot(xs, ys, 'o-', color='black', lw=2)

    # Plot trajectory up to final state
    for i in range(robot.n_links):
        ax0.plot(
            robot.data.pos_x_full[i, :],
            robot.data.pos_y_full[i, :],
            ':',
            lw=1,
        )

    for i in range(robot.n_links):
        ax1.plot(robot.data.t_full[0, :], robot.data.x_full[i, :])

    # y_lim = theta_d * 1.5
    # ax1.set_ylim(-y_lim, y_lim)

    for i in range(robot.n_links):
        ax2.plot(
            robot.data.t_full[0, :],
            robot.data.u_full[i, :],
            label=f'u{i + 1}',
        )
    ax2.legend()

    for i in range(robot.n_links):
        ax3.plot(
            robot.data.t_full[0, :],
            robot.data.tau_full[i, :],
            label=f'τ{i + 1}',
        )
    ax3.legend()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    n_links = 1
    links = {
        'l': [1.0] * n_links,
        'm': [1.0] * n_links,
        'b': [0.2] * n_links,
    }
    robot = PlanarRobotNR(links)

    # robot.simulate(
    #     x0=np.array([0.0] * n_links * 2),
    #     method='velocity',
    #     theta_sp=np.pi / 3,
    #     vel_d=np.pi / 10,
    #     kp=1.0,
    #     ki=0.0,
    #     kd=0.0,
    # )
