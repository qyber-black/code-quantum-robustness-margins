function zeta = differential_sensitivity(H_list, dH_list, dt, Uf, varargin)
%DIFFERENTIAL_SENSITIVITY Gate-fidelity sensitivity zeta at the given point.
%   Uses the product-derivative form:
%     zeta = (1/N) sum_k Re Tr( Uf' * D^{(k)} * exp(-i*phi) )
%   where D^{(k)} inserts dU^{(k)}/dmu into the ordered product.
%
%   Name-value:
%     'method'   'exact' (default) evaluates dU^{(k)}/dmu in closed form in the
%                eigenbasis of H^{(k)}, which is exact for the piecewise-constant
%                controls assumed throughout; 'quadrature' uses Gauss-Legendre.
%     'n_quad'   quadrature nodes, used only by 'quadrature' (default 32)
%
%   A positional n_quad, differential_sensitivity(..., Uf, 32), is accepted;
%   it applies only to the 'quadrature' method.
%
%   See also QROBUSTNESS.DU_DMU_EXACT, QROBUSTNESS.DU_DMU_QUAD.

    opts = qrobustness.parse_dU_options(varargin{:});
    use_exact = strcmp(opts.method, 'exact');

    tau = numel(H_list);
    N = size(H_list{1}, 1);

    Useg = cell(1, tau);
    Vs = cell(1, tau);
    lams = cell(1, tau);
    for k = 1:tau
        if use_exact
            [Vs{k}, lams{k}] = qrobustness.segment_eig(H_list{k});
            Useg{k} = qrobustness.segment_propagator(Vs{k}, lams{k}, dt);
        else
            Useg{k} = expm(-1i * dt * H_list{k});
        end
    end

    % Pref{k} = U{k-1}*...*U{1}, Pref{1}=I
    Pref = cell(1, tau + 1);
    Pref{1} = eye(N);
    for k = 2:tau + 1
        Pref{k} = Useg{k - 1} * Pref{k - 1};
    end
    Utot = Pref{tau + 1};

    F = qrobustness.gate_fidelity(Utot, Uf);
    if F <= 0
        error('qrobustness:zeta:ZeroFid', 'Fidelity must be positive for phase.');
    end
    z = trace(Uf' * Utot);
    e_i_phi = z / abs(z);
    e_minus_i_phi = conj(e_i_phi);

    % Suff{k} = U{tau}*...*U{k}, Suff{tau+1}=I
    Suff = cell(1, tau + 1);
    Suff{tau + 1} = eye(N);
    for k = tau:-1:1
        Suff{k} = Suff{k + 1} * Useg{k};
    end

    if ~use_exact
        [nodes, weights] = qrobustness.gauss_legendre_01(opts.n_quad);
    end

    zeta = 0;
    for k = 1:tau
        if use_exact
            dUk = qrobustness.dU_dmu_exact(Vs{k}, lams{k}, dH_list{k}, dt);
        else
            dUk = qrobustness.dU_dmu_quad(H_list{k}, dH_list{k}, dt, nodes, weights);
        end
        Dk = Suff{k + 1} * dUk * Pref{k};
        zeta = zeta + real(trace(Uf' * Dk * e_minus_i_phi));
    end
    zeta = zeta / N;
end

% SPDX-FileCopyrightText: (C) 2026 F. C. Langbein <frank@langbein.org>
% SPDX-FileCopyrightText: (C) 2026 S. P. O'Neil <sean.oneil@westpoint.edu>
% SPDX-FileCopyrightText: (C) 2026 S. Schirmer <s.m.shermer@gmail.com>
% SPDX-FileCopyrightText: (C) 2026 C. A. Weidner <c.weidner@bristol.ac.uk>
% SPDX-FileCopyrightText: (C) 2026 E. A. Jonckheere <jonckhee@usc.edu>
%
% SPDX-License-Identifier: AGPL-3.0-or-later
