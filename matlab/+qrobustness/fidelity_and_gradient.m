function [F, g1, g2] = fidelity_and_gradient(H0, H1, H2, u1, u2, Uf, dt, varargin)
%FIDELITY_AND_GRADIENT Gate fidelity and GRAPE gradients dF/du1, dF/du2.
%
%   Name-value:
%     'method'   'exact' (default) or 'quadrature'; see
%                QROBUSTNESS.DIFFERENTIAL_SENSITIVITY
%     'n_quad'   quadrature nodes, used only by 'quadrature' (default 32)
%
%   A positional n_quad in argument 8 is accepted; it applies only to the
%   'quadrature' method.
%
%   On the 'exact' path one eigendecomposition per interval serves the
%   propagator and both control derivatives.

    opts = qrobustness.parse_dU_options(varargin{:});
    use_exact = strcmp(opts.method, 'exact');

    u1 = u1(:).';
    u2 = u2(:).';
    tau = numel(u1);
    N = size(H0, 1);

    H_list = cell(1, tau);
    Useg = cell(1, tau);
    Vs = cell(1, tau);
    lams = cell(1, tau);
    for k = 1:tau
        H_list{k} = H0 + u1(k) * H1 + u2(k) * H2;
        if use_exact
            [Vs{k}, lams{k}] = qrobustness.segment_eig(H_list{k});
            Useg{k} = qrobustness.segment_propagator(Vs{k}, lams{k}, dt);
        else
            Useg{k} = expm(-1i * dt * H_list{k});
        end
    end

    Pref = cell(1, tau + 1);
    Pref{1} = eye(N);
    for k = 2:tau + 1
        Pref{k} = Useg{k - 1} * Pref{k - 1};
    end
    Utot = Pref{tau + 1};
    F = qrobustness.gate_fidelity(Utot, Uf);
    if F <= 0
        error('qrobustness:grad:ZeroFid', 'Fidelity must be positive for phase.');
    end
    z = trace(Uf' * Utot);
    e_minus_i_phi = conj(z / abs(z));

    Suff = cell(1, tau + 1);
    Suff{tau + 1} = eye(N);
    for k = tau:-1:1
        Suff{k} = Suff{k + 1} * Useg{k};
    end

    if ~use_exact
        [nodes, weights] = qrobustness.gauss_legendre_01(opts.n_quad);
    end

    g1 = zeros(1, tau);
    g2 = zeros(1, tau);
    for k = 1:tau
        if use_exact
            dUk1 = qrobustness.dU_dmu_exact(Vs{k}, lams{k}, H1, dt);
            dUk2 = qrobustness.dU_dmu_exact(Vs{k}, lams{k}, H2, dt);
        else
            dUk1 = qrobustness.dU_dmu_quad(H_list{k}, H1, dt, nodes, weights);
            dUk2 = qrobustness.dU_dmu_quad(H_list{k}, H2, dt, nodes, weights);
        end
        D1 = Suff{k + 1} * dUk1 * Pref{k};
        D2 = Suff{k + 1} * dUk2 * Pref{k};
        g1(k) = real(trace(Uf' * D1 * e_minus_i_phi)) / N;
        g2(k) = real(trace(Uf' * D2 * e_minus_i_phi)) / N;
    end
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
