import casadi as ca
import matplotlib.animation as animation
import numpy as np
import sympy as sp
from IPython.display import HTML
from matplotlib import gridspec
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from sympy import (
    Eq,
    ImmutableDenseMatrix,
    Matrix,
    Symbol,
    lambdify,
    simplify,
    symbols,
)

G = 9.81


class PlanarRobotNR:
    def __init__(
        self,
        links: dict[str, list[float]],
        n_max: float = 5,
        t_step: float = 0.01,
    ) -> None:
        self.t_step = t_step
        self.n_max = n_max
        self.links = links
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
        self.M_func = lambdify(
            self.q_sym + self.mlb_sym,
            self.M_sym,
            modules='numpy',
        )
        self.C_func = lambdify(
            self.dq_sym + self.q_sym + self.mlb_sym,
            self.C_sym,
            modules='numpy',
        )
        self.G_func = lambdify(
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

    def simulate(self, x0, **kwargs):
        """
        Simulate the dynamics over time using constant torque input.

        Parameters:
            x0 : np.ndarray
                Initial state vector [q1, ..., qn, dq1, ..., dqn]
            tau : np.ndarray
                Constant torque vector [tau1, ..., taun]
        """
        n = len(x0) // 2
        t_vals = np.arange(0, self.n_max + 1) * self.t_step
        state_log = np.zeros((len(t_vals), len(x0)))
        xC_log = np.zeros((len(t_vals), n))
        yC_log = np.zeros((len(t_vals), n))
        tau_log = np.zeros((len(t_vals), n))
        u_log = np.zeros((len(t_vals), n))

        state = np.array(x0, dtype=float)
        state_log[0, :] = state

        dtheta_prev = np.zeros(n)
        theta_e_prev = np.zeros(n)
        theta_i = np.zeros(n)
        for i in range(1, len(t_vals)):
            l = self.links['l']
            m = self.links['m']
            b = self.links['b']

            if kwargs['method'] == 'torque_const':
                tau = kwargs['value']
                u = tau

            elif kwargs['method'] == 'openloop':
                t1 = kwargs['t1']
                t2 = kwargs['t2']
                tf = kwargs['tf']
                theta_d = kwargs['theta_d']
                tau_max = kwargs['tau_max']
                t = i * self.t_step

                vel_d = 2 * theta_d / (tf + t2 - t1)

                if t <= t1:
                    u = vel_d / t1 * t
                elif (t > t1) and (t <= t2):
                    u = vel_d
                elif (t > t2) and (t <= tf):
                    u = vel_d * (tf - t) / (tf - t2)
                else:
                    u = theta_d * 0

                # print(
                #     f't={t}_u={u}_vel_d={vel_d}_ddq={(u - dtheta_prev) / self.t_step}'
                # )
                theta = state[:n]
                tau = self.compute_tau(
                    q=theta,
                    dq=u,
                    ddq=(u - dtheta_prev) / self.t_step,
                    m=m,
                    l=l,
                    b=b,
                )
                tau = np.clip(tau, -tau_max, tau_max)
                dtheta_prev = u
            elif kwargs['method'] == 'velocity':
                theta = state[:n]
                theta_sp = kwargs['theta_sp']
                vel_d = kwargs['vel_d']
                kp = kwargs['kp']
                ki = kwargs['ki']
                kd = kwargs['kd']
                tau_max = kwargs['tau_max']

                theta_e = theta_sp - theta
                u = (
                    vel_d
                    + kp * theta_e
                    + ki * theta_i * self.t_step
                    + kd * (theta_e - theta_e_prev) / self.t_step
                )

                theta_e_prev = theta_e
                theta_i += theta_e

                tau = self.compute_tau(
                    q=theta,
                    dq=u,
                    ddq=(u - dtheta_prev) / self.t_step,
                    m=m,
                    l=l,
                    b=b,
                )
                tau = np.clip(tau, -tau_max, tau_max)
                dtheta_prev = u

            dx, xC, yC = self.dynamics(
                0,
                state,
                tau,
                m,
                l,
                b,
            )
            # print(dx)

            state += self.t_step * dx
            state_log[i, :] = state
            xC_log[i, :] = xC
            yC_log[i, :] = yC
            tau_log[i, :] = tau.flatten()
            u_log[i, :] = u

        print('Finished simulation, generating animation...')

        return {
            't_vals': t_vals,
            'xC_log': xC_log,
            'yC_log': yC_log,
            'x_log': state_log,
            'u_log': u_log,
            'tau_log': tau_log,
        }
        # Lưu kết quả
        ratio = 1
        return animate_all_dynamics(
            t_vals=t_vals[::ratio],
            xC_log=xC_log[::ratio],
            yC_log=yC_log[::ratio],
            x_log=state_log[::ratio],
            tau_log=tau_log[::ratio],  # nếu tau cố định
        )

    def compute_x_u_v(self, q, l, t):
        """
        Compute the symbolic center of mass positions and squared velocities
        for each link in a planar n-link manipulator.

        Parameters:
            q (list): Joint angle functions q1(t), ..., qn(t)
            l (list): Symbols l1, ..., ln for link lengths
            t (sympy.Symbol): Time symbol

        Returns:
            x_c (list): x-coordinates of center of mass
            y_c (list): y-coordinates of center of mass
            v_c_sq (list): squared velocities of center of mass
        """
        x_c, y_c, v_c_sq = [], [], []
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

            x_c.append(xi)
            y_c.append(yi)
            v_c_sq.append(vxi**2 + vyi**2)

        return x_c, y_c, v_c_sq

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
        I = [1 / 3 * m[i] * l[i] ** 2 for i in range(n)]

        # Center of mass positions and velocities
        x_c, y_c, v_c_sq = self.compute_x_u_v(q, l, t)

        # Kinetic energy (translational + rotational)
        T = sum(
            0.5 * m[i] * v_c_sq[i]  # + 0.5 * I[i] * sum(dq[: i + 1]) ** 2
            for i in range(n)
        )

        # Potential energy
        V = sum(m[i] * g * y_c[i] for i in range(n))

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

        return eqs, q, dq, ddq, list(l), list(m), list(b), L, tau, x_c, y_c

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

    def compute_tau(self, q, dq, ddq, m, l, b):
        # print(f'q={q}, dq = {dq}')
        M_val = self.M_func(*q, *m, *l, *b)
        C_val = self.C_func(*q, *dq, *m, *l, *b)
        G_val = self.G_func(*q, *m, *l, *b)

        # print(f'ddq = {ddq}')
        # print(C_val)
        # print(G_val)
        # print(M_val @ ddq.reshape((-1, 1)))
        # print(M_val @ ddq.reshape((-1, 1)) + C_val)

        # print(M_val @ ddq.reshape((-1, 1)) + C_val + G_val)
        return M_val @ ddq.reshape((-1, 1)) + C_val + G_val

    def dynamics(self, t, x, tau, m, l, b):
        n = len(x) // 2
        q = x[:n]
        dq = x[n:]

        M_val = self.M_func(*q, *m, *l, *b)
        C_val = self.C_func(*q, *dq, *m, *l, *b)
        G_val = self.G_func(*q, *m, *l, *b)
        xC_val = self.xC_func(*q, *m, *l, *b)
        yC_val = self.yC_func(*q, *m, *l, *b)

        # print(f'tau = {tau.reshape((n, 1))}')
        # print(f'dq.reshape((n, 1)) = {dq.reshape((n, 1))}')
        # print(tau.reshape((n, 1)))
        # print(tau.reshape((n, 1)) - C_val @ dq.reshape((n, 1)))
        # print(tau.reshape((n, 1)) - C_val @ dq.reshape((n, 1)) - G_val)

        ddq = (
            np.linalg.pinv(M_val)
            @ (tau.reshape((n, 1)) - C_val - G_val).flatten()
        )

        return np.concatenate([dq, ddq]), xC_val, yC_val


def animate_all_dynamics(
    t_vals,
    xC_log,
    yC_log,
    x_log,
    u_log,
    tau_log,
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
    )

    # --- Subplot 1: Forward Kinematics
    ax0 = fig.add_subplot(spec[:, 0])
    ax0.set_xlim(np.min(xC_log) - 0.5, np.max(xC_log) + 0.5)
    ax0.set_ylim(np.min(yC_log) - 0.5, np.max(yC_log) + 0.5)
    ax0.set_aspect('equal')
    ax0.set_title('Forward Kinematics')
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid(True)

    (link_line,) = ax0.plot([], [], 'o-', color='black', lw=2)
    traj_lines = [ax0.plot([], [], '-', lw=1)[0] for _ in range(n)]
    traj_x, traj_y = [[] for _ in range(n)], [[] for _ in range(n)]

    # --- Subplot 2: Joint states x
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.set_title('Joint States')
    ax1.set_ylabel('x / dq')
    ax1.grid(True)
    x_state_lines = [
        ax1.plot([], [], label=f'x{i + 1}')[0] for i in range(x_log.shape[1])
    ]
    ax1.legend()
    ax1.set_xlim(0, t_vals[-1])
    ax1.set_ylim(-2 * np.pi, 2 * np.pi)

    # --- Subplot 3: Control
    ax2 = fig.add_subplot(spec[1, 1])
    ax2.set_title('Control inputs')
    ax2.set_ylabel('Control inputs')
    ax2.grid(True)
    u_state_lines = [
        ax2.plot([], [], label=f'u{i + 1}')[0] for i in range(u_log.shape[1])
    ]
    ax2.legend()
    ax2.set_xlim(0, t_vals[-1])
    ax2.set_ylim(-2 * np.pi, 2 * np.pi)

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
    ax3.set_ylim(-20, 20)

    x_state_hist = [[] for _ in range(x_log.shape[1])]
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
            *x_state_lines,
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

        # x states
        for i in range(x_log.shape[1]):
            x_state_hist[i].append(x_log[frame, i])
            x_state_lines[i].set_data(t_vals[: frame + 1], x_state_hist[i])

        # u states
        for i in range(u_log.shape[1]):
            u_state_hist[i].append(u_log[frame, i])
            u_state_lines[i].set_data(t_vals[: frame + 1], u_state_hist[i])
        # torques
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


def plot_final_kinematics(
    t_vals,
    xC_log,
    yC_log,
    x_log,
    u_log,
    tau_log,
    save_path=None,
):
    T, n = xC_log.shape
    final_frame = -1

    fig = plt.figure(figsize=(12, 6))
    spec = gridspec.GridSpec(
        ncols=2,
        nrows=3,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
    )

    # --- Subplot 1: Forward Kinematics
    ax0 = fig.add_subplot(spec[:, 0])
    ax0.set_xlim(-1.5, 1.5)
    ax0.set_ylim(-1.5, 1.5)
    ax0.set_aspect('equal')
    ax0.set_title('Forward Kinematics (Final State)')
    ax0.set_xlabel('x [m]')
    ax0.set_ylabel('y [m]')
    ax0.grid(True)

    # Plot robot links
    xs = [0, *list(xC_log[final_frame])]
    ys = [0, *list(yC_log[final_frame])]
    ax0.plot(xs, ys, 'o-', color='black', lw=2)

    # Plot trajectory up to final state
    for i in range(n):
        ax0.plot(xC_log[:T, i], yC_log[:T, i], '-', lw=1)

    # --- Subplot 2: Joint states x
    ax1 = fig.add_subplot(spec[0, 1])
    ax1.set_title('Joint States')
    ax1.set_ylabel('x / dq')
    ax1.grid(True)

    n = x_log.shape[1] // 2
    for i in range(x_log.shape[1]):
        if i < n:
            ax1.plot(t_vals, x_log[:, i], label=f'theta{i + 1}')
    ax1.legend()
    ax1.set_xlim(0, t_vals[-1])

    y_lim = np.max(np.abs(x_log[:, :n])) * 1.1
    ax1.set_ylim(-y_lim, y_lim)

    # --- Subplot 3: Control
    ax2 = fig.add_subplot(spec[1, 1], sharex=ax1)
    ax2.set_title('Control input')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Control input [Nm]')
    ax2.grid(True)
    for i in range(u_log.shape[1]):
        ax2.plot(t_vals, u_log[:, i], label=f'u{i + 1}')
    ax2.legend()
    ax2.set_xlim(0, t_vals[-1])
    y_lim = np.max(np.abs(u_log)) * 1.1
    ax2.set_ylim(-y_lim, y_lim)

    # --- Subplot 4: Torques τ
    ax3 = fig.add_subplot(spec[2, 1], sharex=ax1)
    ax3.set_title('Torques')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Torque [Nm]')
    ax3.grid(True)
    for i in range(tau_log.shape[1]):
        ax3.plot(t_vals, tau_log[:, i], label=f'τ{i + 1}')
    ax3.legend()
    ax3.set_xlim(0, t_vals[-1])
    y_lim = np.max(np.abs(tau_log)) * 1.1
    ax3.set_ylim(-y_lim, y_lim)

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

    robot.simulate(
        x0=np.array([0.0] * n_links * 2),
        method='velocity',
        theta_sp=np.pi / 3,
        vel_d=np.pi / 10,
        kp=1.0,
        ki=0.0,
        kd=0.0,
    )
