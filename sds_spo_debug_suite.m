function OUT = sds_spo_debug_suite(expType, userCfg)
%==========================================================================
% SDS + SPO+ DEBUG SUITE (STRICT TWO-STAGE VERSION)
%
%   expType = 'v1' : low-affine-dimension warm-up
%   expType = 'v2' : full-dimensional Euclidean-ball-prior experiment
%
% STRICT TWO-STAGE AT EACH PREFIX t:
%   For each t = 1,2,...,nTrain:
%
%   Stage I:
%       - fit centered OLS on the first t labeled samples
%       - generate pseudo-costs on the first t fresh discovery contexts
%       - run Algorithm-2-like cumulative learner from scratch
%       - freeze the learned subspace Uhat_t
%
%   Stage II:
%       - train FULL SPO+ from scratch on the first t labeled samples
%       - train COMPRESSED SPO+ from scratch on the first t labeled samples
%         with Uhat_t fixed
%
% This is therefore a STRICT two-stage implementation.
%
% IMPORTANT HONEST NOTE
%   Stage I below is NOT the exact FI/facet-hit Algorithm 1/2 from the paper.
%   It is an oracle-based approximate separation version:
%
%       current D  --warm start-->
%       pseudo-cost c_hat
%       search for witness costs inside the current measurement fiber
%       if a witness changes the optimal path, append x_w - x_ref
%
%   So it is much closer to Algorithm 2 than an SVD basis learner, but it
%   is still an approximation.
%
% IMPORTANT NO-LEAK FIX
%   Stage II and evaluation DO NOT access:
%       - prob.prior
%       - prob.Ustar
%       - corridorEdges
%       - affine-hull information
%
%   They only access:
%       - c0
%       - predRadius
%
%   Hence the full baseline is not secretly projected onto the true affine
%   hull, and the compressed baseline is not secretly corrected by the true
%   subspace either.
%
% Usage:
%   R1 = sds_spo_debug_suite('v1');
%   R2 = sds_spo_debug_suite('v2');
%
% Example override:
%   cfg = struct();
%   cfg.g = 20;
%   cfg.p = 8;
%   cfg.nTrain = 150;
%   cfg.nTrial = 2;
%   cfg.data = struct('dstarTarget', 8);
%   cfg.stage1 = struct('numPasses', 1, 'numRandomFiberDirs', 16);
%   cfg.stage2 = struct('numEpochsFull', 3, 'numEpochsRed', 3, 'batchSize', 32);
%   R = sds_spo_debug_suite('v2', cfg);
%==========================================================================

if nargin < 1 || isempty(expType)
    expType = 'v2';
end
if nargin < 2 || isempty(userCfg)
    userCfg = struct();
end

cfg = default_cfg(expType);
cfg = merge_struct(cfg, userCfg);

% Keep Stage-II prediction radius synced to data prior radius unless
% the user explicitly overrides stage2.predRadius.
if ~isfield(userCfg, 'stage2') || ~isfield(userCfg.stage2, 'predRadius')
    cfg.stage2.predRadius = cfg.data.priorRadius;
end

validate_cfg(cfg);
rng(cfg.seed, 'twister');

edge = build_grid_edge_maps(cfg.g);
prob = build_problem(cfg, edge);

stage2Spec = struct();
stage2Spec.c0 = prob.c0;
stage2Spec.predRadius = cfg.stage2.predRadius;

fprintf('\n============================================================\n');
fprintf('Running STRICT two-stage SDS/SPO debug suite | expType = %s\n', upper(cfg.expType));
fprintf('Grid g = %d | ambient d = %d | target d* = %d | p = %d\n', ...
    cfg.g, edge.d, prob.trueDstar, cfg.p);
fprintf('nTrain = %d | nTest = %d | nTrial = %d\n', ...
    cfg.nTrain, cfg.nTest, cfg.nTrial);
fprintf('Stage I: ridge = %.2e | passes = %d | maxDiscoveryPerT = %s\n', ...
    cfg.stage1.ridge, cfg.stage1.numPasses, num2str_inf(cfg.stage1.maxDiscoveryPerT));
fprintf('         randFiberDirs = %d | maxAdds/sample = %d\n', ...
    cfg.stage1.numRandomFiberDirs, cfg.stage1.maxAddsPerSample);
fprintf('Stage II: numEpochsFull = %d | numEpochsRed = %d | batchSize = %d | predRadius = %.3f\n', ...
    cfg.stage2.numEpochsFull, cfg.stage2.numEpochsRed, cfg.stage2.batchSize, cfg.stage2.predRadius);
fprintf('============================================================\n\n');

OUT = run_all_trials(cfg, prob, edge, stage2Spec);

end

%==========================================================================
% DEFAULT CONFIG
%==========================================================================

function cfg = default_cfg(expType)

cfg = struct();
cfg.expType = lower(char(expType));

% ----------------------------- global ------------------------------------
cfg.seed         = 20260313;
cfg.g            = 20;       % number of nodes per side
cfg.p            = 8;        % choose >= dstarTarget if you want all d* activated
cfg.nTrain       = 300;      % strict two-stage at every prefix can be expensive
cfg.nTest        = 300;
cfg.nTrial       = 3;
cfg.verboseEvery = 25;

cfg.plot = struct();
cfg.plot.dimPlotMax  = 12;      % only show dim(W) in first few iterations
cfg.plot.saveFigures = false;
cfg.plot.resultsDir  = 'results';

% ----------------------------- data --------------------------------------
cfg.data = struct();

cfg.data.dstarTarget  = 8;      % target decision-relevant dimension
cfg.data.baselineLow  = 10;
cfg.data.baselineHigh = 100;

% prior radius:
%   v1: latent affine-ball radius
%   v2: Euclidean ball radius around c0
cfg.data.priorRadius = 0.90;

% true linear conditional mean:
%   mu(x) = c0 + U* A x
% with bounded contexts x in [-1,1]^p and row-orthonormal A
cfg.data.muScale = 0.45;

% bounded zero-mean latent noise
cfg.data.latentNoiseScale = 0.15;

% extra zero-mean orthogonal nuisance noise for v2 only
cfg.data.nuisanceScale = 0.10;

% ----------------------------- stage I -----------------------------------
cfg.stage1 = struct();

% centered OLS
cfg.stage1.ridge = 1e-6;

% strict two-stage Stage-I controls
cfg.stage1.numPasses         = 1;     % passes over the pseudo-cost stream
cfg.stage1.maxDiscoveryPerT  = inf;   % use first min(t, maxDiscoveryPerT) discovery contexts

% Algorithm-2-like cumulative learner
cfg.stage1.numRandomFiberDirs = 24;
cfg.stage1.maxAddsPerSample   = 2;
cfg.stage1.alphaList          = [1.00, 0.50, 0.25];
cfg.stage1.linDepTol          = 1e-8;

% optional cap on learned basis size
cfg.stage1.maxBasisDim = inf;

% ----------------------------- stage II ----------------------------------
cfg.stage2 = struct();

cfg.stage2.lrFull = 0.040;
cfg.stage2.lrRed  = 0.060;

cfg.stage2.numEpochsFull = 3;
cfg.stage2.numEpochsRed  = 3;
cfg.stage2.batchSize     = 32;
cfg.stage2.gradClip      = 5.0;

% IMPORTANT:
%   This is the only constraint Stage II sees.
%   It is an ambient Euclidean ball around c0, NOT the data prior projector.
cfg.stage2.predRadius = cfg.data.priorRadius;

% ------------------------- experiment-specific ---------------------------
switch cfg.expType
    case 'v1'
        cfg.p = max(cfg.p, cfg.data.dstarTarget);

    case 'v2'
        cfg.p = max(cfg.p, cfg.data.dstarTarget);

    otherwise
        error('Unknown expType "%s". Use "v1" or "v2".', cfg.expType);
end

end

function validate_cfg(cfg)

if ~(strcmp(cfg.expType, 'v1') || strcmp(cfg.expType, 'v2'))
    error('cfg.expType must be "v1" or "v2".');
end

if cfg.g < 4
    error('Need g >= 4.');
end

maxGadgets = floor((cfg.g - 1) / 2);
if cfg.data.dstarTarget > maxGadgets
    error(['For the current diagonal gadget construction, dstarTarget <= floor((g-1)/2). ' ...
           'Current max = %d, requested = %d.'], maxGadgets, cfg.data.dstarTarget);
end

if cfg.p < cfg.data.dstarTarget
    error('Need p >= dstarTarget if you want the contextual model to excite all d* directions.');
end

if cfg.data.muScale + cfg.data.latentNoiseScale >= cfg.data.priorRadius
    error('Need muScale + latentNoiseScale < priorRadius.');
end

if strcmp(cfg.expType, 'v2')
    if cfg.data.muScale + cfg.data.latentNoiseScale + cfg.data.nuisanceScale >= cfg.data.priorRadius
        error('Need muScale + latentNoiseScale + nuisanceScale < priorRadius for v2.');
    end
end

if cfg.stage2.predRadius <= 0
    error('stage2.predRadius must be positive.');
end

if cfg.stage2.batchSize < 1
    error('stage2.batchSize must be >= 1.');
end

if cfg.stage1.numPasses < 1
    error('stage1.numPasses must be >= 1.');
end

end

%==========================================================================
% MAIN DRIVER
%==========================================================================

function OUT = run_all_trials(cfg, prob, edge, stage2Spec)

T = cfg.nTrain;

riskFull = zeros(cfg.nTrial, T);
riskRed  = zeros(cfg.nTrial, T);
dimHist  = zeros(cfg.nTrial, T);
hardHist = zeros(cfg.nTrial, T);

trialDetail = cell(cfg.nTrial, 1);

for tr = 1:cfg.nTrial
    fprintf('=== Trial %d / %d ===\n', tr, cfg.nTrial);
    rng(cfg.seed + 1000 * tr, 'twister');

    % ---------- build one linear ground-truth contextual model ----------
    Atrue = make_full_row_rank_rows(prob.trueDstar, cfg.p);      % d* x p
    Atrue = (cfg.data.muScale / sqrt(cfg.p)) * Atrue;            % bounded linear mean

    % ---------- data ----------
    Xlab  = sample_contexts(cfg.nTrain, cfg.p);
    Xdisc = sample_contexts(cfg.nTrain, cfg.p);   % fresh discovery contexts
    Xte   = sample_contexts(cfg.nTest,  cfg.p);

    Clab = generate_costs_from_contexts(Xlab, Atrue, prob, cfg);
    Cte  = generate_costs_from_contexts(Xte,  Atrue, prob, cfg);

    dimPerIter  = zeros(1, T);
    addPerIter  = zeros(1, T);

    for t = 1:T
        % ================================================================
        % STRICT STAGE I:
        %   learn Uhat_t from scratch using only the prefix 1:t
        % ================================================================
        [Uhat, infoStage1] = learn_subspace_two_stage( ...
            Xlab(1:t,:), Clab(1:t,:), Xdisc(1:t,:), prob, edge, cfg.stage1);

        dimPerIter(t) = size(Uhat, 2);
        addPerIter(t) = infoStage1.totalAdded;

        % ================================================================
        % STRICT STAGE II:
        %   with Uhat_t FIXED, train full / compressed models from scratch
        %   on the same prefix 1:t
        % ================================================================
        Bfull = train_full_model_two_stage( ...
            Xlab(1:t,:), Clab(1:t,:), stage2Spec, edge, cfg.stage2);

        Gred = train_reduced_model_two_stage( ...
            Xlab(1:t,:), Clab(1:t,:), Uhat, stage2Spec, edge, cfg.stage2);

        % ================================================================
        % Evaluate at EVERY iteration
        % ================================================================
        riskFull(tr, t) = mean_spo_risk_full(Bfull, Xte, Cte, stage2Spec, edge);
        riskRed(tr,  t) = mean_spo_risk_red(Gred, Uhat, Xte, Cte, stage2Spec, edge);

        if mod(t, cfg.verboseEvery) == 0 || t <= 10 || t == T
            fprintf('  t=%3d | dim(W)=%2d | stage1Adds=%2d | riskF=%.4g | riskR=%.4g\n', ...
                t, dimPerIter(t), addPerIter(t), riskFull(tr,t), riskRed(tr,t));
        end
    end

    hardHist(tr,:) = addPerIter;
    dimHist(tr,:)  = dimPerIter;

    detail = struct();
    detail.Atrue = Atrue;
    detail.Xlab = Xlab;
    detail.Clab = Clab;
    detail.Xdisc = Xdisc;
    detail.Xte = Xte;
    detail.Cte = Cte;
    detail.dimPerIter = dimPerIter;
    detail.stage1AddPerIter = addPerIter;
    trialDetail{tr} = detail;

    fprintf('\n');
end

% ------------------------- aggregate statistics ---------------------------
eps0 = 1e-12;

meanLogFull = mean(log10(riskFull + eps0), 1, 'omitnan');
meanLogRed  = mean(log10(riskRed  + eps0), 1, 'omitnan');

[~, ciLogFull] = mean_ci90(log10(riskFull + eps0));
[~, ciLogRed]  = mean_ci90(log10(riskRed  + eps0));

meanDim = mean(dimHist, 1, 'omitnan');

riskRatio = riskFull ./ max(riskRed, eps0);
meanRiskRatio = mean(riskRatio, 1, 'omitnan');
[~, ciRiskRatio] = mean_ci90(riskRatio);

paramRatioTrue = edge.d / prob.trueDstar;
boundRatioTrue = sqrt(edge.d / prob.trueDstar);

fprintf('------------------------------------------------------------\n');
fprintf('ambient d                    = %d\n', edge.d);
fprintf('true d*                      = %d\n', prob.trueDstar);
fprintf('parameter ratio d/d*         = %.4f\n', paramRatioTrue);
fprintf('bound-term ratio sqrt(d/d*)  = %.4f\n', boundRatioTrue);
fprintf('empirical risk ratio at final iter (Full/Reduced) = %.4f +/- %.4f (90%% CI)\n', ...
    meanRiskRatio(end), ciRiskRatio(end));
fprintf('mean final learned dim(W)    = %.4f\n', meanDim(end));
fprintf('------------------------------------------------------------\n');

% ------------------------------- plots -----------------------------------
make_plots(cfg, prob, edge, riskFull, riskRed, dimHist);

% ------------------------------- output ----------------------------------
OUT = struct();
OUT.cfg = cfg;
OUT.prob = prob;
OUT.edge = edge;
OUT.stage2Spec = stage2Spec;

OUT.riskFull = riskFull;
OUT.riskRed  = riskRed;
OUT.dimHist  = dimHist;
OUT.hardHist = hardHist;
OUT.trialDetail = trialDetail;

OUT.meanLogFull = meanLogFull;
OUT.meanLogRed  = meanLogRed;
OUT.ciLogFull   = ciLogFull;
OUT.ciLogRed    = ciLogRed;
OUT.meanDim     = meanDim;

OUT.riskRatio      = riskRatio;
OUT.meanRiskRatio  = meanRiskRatio;
OUT.ciRiskRatio    = ciRiskRatio;
OUT.paramRatioTrue = paramRatioTrue;
OUT.boundRatioTrue = boundRatioTrue;

end

%==========================================================================
% BUILD EXPERIMENT PROBLEM
%==========================================================================

function prob = build_problem(cfg, edge)

switch cfg.expType
    case 'v1'
        % Low-affine-dimension warm-up:
        %   C = { c0 + U*z : ||z|| <= rho }, so affdim(C) = d*
        [Ustar, gadgetInfo] = build_diagonal_switch_basis(edge, cfg.data.dstarTarget);

        c0 = cfg.data.baselineLow * ones(edge.d, 1);

        prior = struct();
        prior.kind   = 'affine_ball';
        prior.c0     = c0;
        prior.Uaff   = Ustar;                 % used only by DATA + Stage I
        prior.radius = cfg.data.priorRadius;

        prob = struct();
        prob.name = 'Version 1';
        prob.c0 = c0;
        prob.prior = prior;
        prob.Ustar = Ustar;
        prob.trueDstar = size(Ustar, 2);
        prob.gadgetInfo = gadgetInfo;

    case 'v2'
        % Full-dimensional Euclidean-ball prior:
        %   C = { c : ||c-c0||_2 <= rho }, affdim(C) = d
        % c0 is corridor-like so all shortest paths stay in a narrow family.
        [corrInfo, corridorEdges, Ustar] = build_diagonal_corridor(edge, cfg.data.dstarTarget);

        c0 = cfg.data.baselineHigh * ones(edge.d, 1);
        c0(corridorEdges) = cfg.data.baselineLow;

        prior = struct();
        prior.kind   = 'euclid_ball';
        prior.c0     = c0;
        prior.radius = cfg.data.priorRadius;

        prob = struct();
        prob.name = 'Version 2';
        prob.c0 = c0;
        prob.prior = prior;
        prob.Ustar = Ustar;
        prob.trueDstar = size(Ustar, 2);
        prob.corridorEdges = corridorEdges;
        prob.corrInfo = corrInfo;

        verify_corridor_margin(cfg, edge);

    otherwise
        error('Unknown expType.');
end

end

function verify_corridor_margin(cfg, edge)
L = 2 * (cfg.g - 1);

maxCorrCost = L * (cfg.data.baselineLow  + cfg.data.priorRadius);
minOneOutCost = (cfg.data.baselineHigh - cfg.data.priorRadius) + ...
                (L - 1) * (cfg.data.baselineLow - cfg.data.priorRadius);

if minOneOutCost <= maxCorrCost
    warning(['The baseline gap may be too small for guaranteed corridor dominance. ' ...
             'Increase baselineHigh or reduce priorRadius.']);
end
end

%==========================================================================
% DATA GENERATION
%==========================================================================

function X = sample_contexts(n, p)
% bounded contexts so the conditional mean remains exactly linear
X = 2 * rand(n, p) - 1;   % Uniform[-1,1]^p
end

function C = generate_costs_from_contexts(X, Atrue, prob, cfg)
% TRUE MODEL:
%   mu(x) = c0 + U_star * (Atrue * x)
% plus zero-mean noise that preserves linear conditional mean.
%
% Because we enforce
%   muScale + latentNoiseScale (+ nuisanceScale) < priorRadius,
% the projection to the prior should almost never be active in normal runs.
% Hence the conditional mean is effectively linear.

n = size(X,1);
d = numel(prob.c0);
r = prob.trueDstar;

C = zeros(n, d);

for i = 1:n
    xi = X(i,:)';

    zMean = Atrue * xi;   % linear in context, bounded in norm by muScale

    % zero-mean bounded latent noise, norm <= latentNoiseScale
    zNoise = (cfg.data.latentNoiseScale / sqrt(r)) * (2 * rand(r,1) - 1);

    if strcmp(prob.name, 'Version 1')
        c = prob.c0 + prob.Ustar * (zMean + zNoise);

    elseif strcmp(prob.name, 'Version 2')
        % zero-mean orthogonal nuisance noise with norm <= nuisanceScale
        v = randn(d,1);
        v = v - prob.Ustar * (prob.Ustar' * v);
        nv = norm(v);
        if nv > 1e-12
            v = v / nv;
        else
            v = zeros(d,1);
        end
        orthNoise = cfg.data.nuisanceScale * (2 * rand - 1) * v;

        c = prob.c0 + prob.Ustar * (zMean + zNoise) + orthNoise;

    else
        error('Unknown problem name.');
    end

    c = project_to_prior(c, prob.prior);
    C(i,:) = c.';
end
end

function A = make_full_row_rank_rows(r, p)
if p < r
    error('Need p >= r.');
end

A = randn(r, p);
while rank(A, 1e-10) < r
    A = randn(r, p);
end

% row-orthonormalize
[Q, ~] = qr(A.', 0);   % p x r
A = Q.';               % r x p
end

%==========================================================================
% STRICT STAGE I
%==========================================================================

function [Uhat, info] = learn_subspace_two_stage(Xlab, Clab, Xdisc, prob, edge, s1)
% STRICT TWO-STAGE Stage I at a fixed sample size t:
%   - fit centered OLS on (Xlab, Clab)
%   - form pseudo-costs on Xdisc
%   - run Algorithm-2-like cumulative learner FROM SCRATCH
%   - return the frozen learned subspace Uhat

Ahat = fit_centered_ols(Xlab, Clab, prob.c0, s1.ridge);

nDiscAvail = size(Xdisc, 1);
nDiscUse = min(nDiscAvail, s1.maxDiscoveryPerT);

D = sparse(edge.d, 0);
totalAdded = 0;

for pass = 1:s1.numPasses
    for j = 1:nDiscUse
        cPseudo = prob.c0 + Ahat * Xdisc(j,:)';
        cPseudo = project_to_prior(cPseudo, prob.prior);

        [D, info1] = stage1_update_alg2like(cPseudo, D, prob.prior, edge, s1);
        totalAdded = totalAdded + info1.numAdded;

        if size(D,2) >= s1.maxBasisDim
            break;
        end
    end

    if size(D,2) >= s1.maxBasisDim
        break;
    end
end

Uhat = orth(full(D));

info = struct();
info.totalAdded = totalAdded;
info.dim = size(Uhat, 2);
info.Ahat = Ahat;

end

function Ahat = fit_centered_ols(X, C, c0, ridge)
% centered multi-response OLS:
%   c - c0 = A_mu * x + eps

[~, p] = size(X);
d = size(C, 2);

if isempty(X)
    Ahat = zeros(d, p);
    return;
end

Y = C - c0.';
XtX = X.' * X + ridge * eye(p);
Ahat = (Y.' * X) / XtX;   % d x p
end

function [D, info] = stage1_update_alg2like(cPseudo, D, prior, edge, s1)
% Algorithm-2-like cumulative learner:
%   warm-start from current D
%   run a pointwise routine on cPseudo
%   append new linearly independent path-difference directions if found
%
% This is NOT the exact FI/facet-hit routine; it is an oracle-based
% approximate separation version.

info = struct();
info.numAdded = 0;
info.hard = false;

for kk = 1:s1.maxAddsPerSample
    if size(D,2) >= s1.maxBasisDim
        break;
    end

    [added, qNew] = pointwise_one_addition(cPseudo, D, prior, edge, s1);
    if ~added
        break;
    end

    D = [D, sparse(qNew)]; %#ok<AGROW>
    info.numAdded = info.numAdded + 1;
    info.hard = true;
end
end

function [added, qNew] = pointwise_one_addition(cRef, D, prior, edge, s1)
% One approximate pointwise update:
%   keep the reference optimal path x0 = x*(cRef)
%   search within the measurement fiber for witness costs
%   if some witness changes the optimal path, add q = x_w - x0

added = false;
qNew = [];

[~, x0] = oracle_monotone_path_dp(cRef, edge);
x0 = full(x0);

cands = sample_fiber_witnesses(cRef, D, prior, s1);

for j = 1:size(cands, 2)
    cw = cands(:,j);

    [~, xw] = oracle_monotone_path_dp(cw, edge);
    xw = full(xw);

    q = xw - x0;
    if norm(q, 2) < 1e-12
        continue;
    end

    if independent_from_span(q, D, s1.linDepTol)
        added = true;
        qNew = q;
        return;
    end
end
end

function cands = sample_fiber_witnesses(cRef, D, prior, s1)

switch prior.kind
    case 'euclid_ball'
        cands = sample_fiber_witnesses_euclid_ball(cRef, D, prior, s1);

    case 'affine_ball'
        cands = sample_fiber_witnesses_affine_ball(cRef, D, prior, s1);

    otherwise
        error('Unknown prior.kind "%s".', prior.kind);
end
end

function cands = sample_fiber_witnesses_euclid_ball(cRef, D, prior, s1)

d = numel(cRef);
cands = zeros(d, 0);

Uq = orth(full(D));   % span of current query directions

for kk = 1:s1.numRandomFiberDirs
    r = randn(d,1);
    if ~isempty(Uq)
        r = r - Uq * (Uq.' * r);  % keep q^T c unchanged for all q in D
    end
    nr = norm(r);
    if nr < 1e-12
        continue;
    end
    delta = r / nr;

    tauPlus  = max_step_euclid_ball(cRef,  delta, prior);
    tauMinus = max_step_euclid_ball(cRef, -delta, prior);

    for a = s1.alphaList
        if tauPlus > 1e-12
            cands(:,end+1) = cRef + a * tauPlus * delta; %#ok<AGROW>
        end
        if tauMinus > 1e-12
            cands(:,end+1) = cRef - a * tauMinus * delta; %#ok<AGROW>
        end
    end
end

cands(:,end+1) = cRef;
end

function cands = sample_fiber_witnesses_affine_ball(cRef, D, prior, s1)

U = prior.Uaff;
zc = U.' * (cRef - prior.c0);   % latent coordinate in affine hull
rAff = size(U, 2);

cands = zeros(size(U,1), 0);

M = full(D).' * U;              % t x rAff
Ur = orth(M.');                 % basis for row-space of M inside R^{rAff}

for kk = 1:s1.numRandomFiberDirs
    zdir = randn(rAff, 1);
    if ~isempty(Ur)
        zdir = zdir - Ur * (Ur.' * zdir);  % preserve all q^T c measurements
    end
    nz = norm(zdir);
    if nz < 1e-12
        continue;
    end
    zdir = zdir / nz;

    tauPlus  = max_step_latent_ball(zc,  zdir, prior.radius);
    tauMinus = max_step_latent_ball(zc, -zdir, prior.radius);

    for a = s1.alphaList
        if tauPlus > 1e-12
            cands(:,end+1) = prior.c0 + U * (zc + a * tauPlus * zdir); %#ok<AGROW>
        end
        if tauMinus > 1e-12
            cands(:,end+1) = prior.c0 + U * (zc - a * tauMinus * zdir); %#ok<AGROW>
        end
    end
end

cands(:,end+1) = cRef;
end

function tau = max_step_euclid_ball(c, delta, prior)
u = c - prior.c0;

a = delta.' * delta;
b = 2 * (u.' * delta);
cc = (u.' * u) - prior.radius^2;

disc = max(b^2 - 4 * a * cc, 0);
tau = max(0, (-b + sqrt(disc)) / (2 * a));
end

function tau = max_step_latent_ball(z, dz, rho)
a = dz.' * dz;
b = 2 * (z.' * dz);
cc = (z.' * z) - rho^2;

disc = max(b^2 - 4 * a * cc, 0);
tau = max(0, (-b + sqrt(disc)) / (2 * a));
end

function tf = independent_from_span(q, D, tol)
q = full(q);

if isempty(D)
    tf = norm(q,2) > tol;
    return;
end

U = orth(full(D));
res = q - U * (U.' * q);
tf = norm(res, 2) > tol * max(1, norm(q,2));
end

%==========================================================================
% STRICT STAGE II
%==========================================================================

function Bfull = train_full_model_two_stage(X, C, stage2Spec, edge, s2)
% Train FULL SPO+ from scratch on the given prefix dataset.
% NO-LEAK: only c0 and predRadius are used.

[n, p] = size(X);
Bfull = zeros(edge.d, p + 1);

if n == 0
    return;
end

for ep = 1:s2.numEpochsFull
    eta = s2.lrFull / sqrt(ep);
    perm = randperm(n);

    for s = 1:s2.batchSize:n
        ids = perm(s : min(s + s2.batchSize - 1, n));
        Grad = zeros(edge.d, p + 1);

        for kk = 1:numel(ids)
            i = ids(kk);
            phi = [X(i,:)'; 1];
            c   = C(i,:)';

            chat = predict_full_noleak(Bfull, phi, stage2Spec);
            v = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + full(v) * phi.';
        end

        Grad = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > s2.gradClip
            Grad = Grad * (s2.gradClip / gnorm);
        end

        Bfull = Bfull - eta * Grad;
    end
end
end

function Gred = train_reduced_model_two_stage(X, C, Uhat, stage2Spec, edge, s2)
% Train COMPRESSED SPO+ from scratch on the given prefix dataset,
% with Uhat fixed throughout Stage II.
% NO-LEAK: no access to prob.prior / Ustar / corridor information.

[~, p] = size(X);

if isempty(Uhat)
    Gred = zeros(0, p + 1);
    return;
end

n = size(X,1);
r = size(Uhat,2);

Gred = zeros(r, p + 1);

if n == 0
    return;
end

for ep = 1:s2.numEpochsRed
    eta = s2.lrRed / sqrt(ep);
    perm = randperm(n);

    for s = 1:s2.batchSize:n
        ids = perm(s : min(s + s2.batchSize - 1, n));
        Grad = zeros(r, p + 1);

        for kk = 1:numel(ids)
            i = ids(kk);
            phi = [X(i,:)'; 1];
            c   = C(i,:)';

            chat = predict_red_noleak(Gred, Uhat, phi, stage2Spec);
            v = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + (Uhat.' * full(v)) * phi.';
        end

        Grad = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > s2.gradClip
            Grad = Grad * (s2.gradClip / gnorm);
        end

        Gred = Gred - eta * Grad;
    end
end
end

function chat = predict_full_noleak(Bfull, phi, stage2Spec)
% Full ambient model constrained only by an ambient Euclidean ball around c0.
% No Ustar / no affine hull / no corridor information is used.

delta = Bfull * phi;
delta = project_ball_centered(delta, stage2Spec.predRadius);
chat  = stage2Spec.c0 + delta;
end

function chat = predict_red_noleak(Gred, Uhat, phi, stage2Spec)
% Reduced model uses ONLY the learned subspace Uhat.
% Again: no true-Ustar projector and no data-prior projector.

if isempty(Uhat) || isempty(Gred)
    chat = stage2Spec.c0;
    return;
end

s = Gred * phi;
s = project_ball_centered(s, stage2Spec.predRadius);

% Since Uhat is orthonormal, ||Uhat*s||_2 = ||s||_2 <= predRadius.
chat = stage2Spec.c0 + Uhat * s;
end

function y = project_ball_centered(x, radius)
nx = norm(x, 2);
if nx <= radius || nx < 1e-12
    y = x;
else
    y = (radius / nx) * x;
end
end

%==========================================================================
% RISK / LOSS / ORACLE
%==========================================================================

function risk = mean_spo_risk_full(Bfull, X, C, stage2Spec, edge)

n = size(X,1);
tot = 0;

for i = 1:n
    phi  = [X(i,:)'; 1];
    c    = C(i,:)';
    chat = predict_full_noleak(Bfull, phi, stage2Spec);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_red(Gred, Uhat, X, C, stage2Spec, edge)

n = size(X,1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c   = C(i,:)';

    chat = predict_red_noleak(Gred, Uhat, phi, stage2Spec);
    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function loss = spo_loss_dp(chat, ctrue, edge)
[~, xHat] = oracle_monotone_path_dp(chat,  edge);
[~, xOpt] = oracle_monotone_path_dp(ctrue, edge);
loss = ctrue.' * (xHat - xOpt);
end

function v = spoplus_subgrad_dp(chat, ctrue, edge)
[~, x0] = oracle_monotone_path_dp(ctrue, edge);
[~, x1] = oracle_monotone_path_dp(2 * chat - ctrue, edge);
v = 2 * (x0 - x1);
end

function [bestCost, x] = oracle_monotone_path_dp(c, edge)
% monotone shortest path by dynamic programming on the DAG

g = edge.g;
h = edge.h;
v = edge.v;

D = inf(g, g);
P = zeros(g, g, 'uint8');  % 1 = from left, 2 = from up
D(1,1) = 0;

for i = 1:g
    for j = 1:g
        cur = D(i,j);
        if isinf(cur)
            continue;
        end

        if j < g
            cand = cur + c(h(i,j));
            if cand < D(i, j+1)
                D(i, j+1) = cand;
                P(i, j+1) = 1;
            end
        end

        if i < g
            cand = cur + c(v(i,j));
            if cand < D(i+1, j)
                D(i+1, j) = cand;
                P(i+1, j) = 2;
            end
        end
    end
end

bestCost = D(g,g);
x = sparse(edge.d, 1);

i = g;
j = g;
while (i > 1) || (j > 1)
    if P(i,j) == 1
        e = h(i, j-1);
        x(e) = 1;
        j = j - 1;
    elseif P(i,j) == 2
        e = v(i-1, j);
        x(e) = 1;
        i = i - 1;
    else
        error('Path reconstruction failed.');
    end
end
end

%==========================================================================
% PRIORS / PROJECTIONS
%==========================================================================

function cProj = project_to_prior(c, prior)
%==========================================================================
% DATA / STAGE-I ONLY
%
% DO NOT use this in Stage II.
%
% In Version 1, prior.kind = 'affine_ball' contains prior.Uaff = Ustar.
% If Stage II calls this projector, the predictor is secretly projected
% onto the true affine hull, which leaks the true subspace.
%==========================================================================

switch prior.kind
    case 'euclid_ball'
        u = c - prior.c0;
        nu = norm(u, 2);
        if nu <= prior.radius
            cProj = c;
        else
            cProj = prior.c0 + (prior.radius / nu) * u;
        end

    case 'affine_ball'
        z = prior.Uaff.' * (c - prior.c0);
        nz = norm(z, 2);
        if nz > prior.radius
            z = (prior.radius / nz) * z;
        end
        cProj = prior.c0 + prior.Uaff * z;

    otherwise
        error('Unknown prior kind.');
end

end

%==========================================================================
% GRID GEOMETRY
%==========================================================================

function edge = build_grid_edge_maps(g)

h = zeros(g, g-1);
idx = 1;
for i = 1:g
    for j = 1:(g-1)
        h(i,j) = idx;
        idx = idx + 1;
    end
end

v = zeros(g-1, g);
for i = 1:(g-1)
    for j = 1:g
        v(i,j) = idx;
        idx = idx + 1;
    end
end

edge = struct();
edge.h = h;
edge.v = v;
edge.g = g;
edge.d = idx - 1;
end

function [Ustar, info] = build_diagonal_switch_basis(edge, m)
% m disjoint 2x2 switch gadgets on the diagonal

Q = zeros(edge.d, m);
squareTL = zeros(m, 2);

for k = 1:m
    i = 2*k - 1;
    j = 2*k - 1;

    squareTL(k,:) = [i, j];

    q = zeros(edge.d, 1);
    q(edge.h(i,   j)) =  1;
    q(edge.v(i, j+1)) =  1;
    q(edge.v(i,   j)) = -1;
    q(edge.h(i+1, j)) = -1;

    Q(:,k) = q / norm(q);
end

Ustar = Q;
info = struct();
info.squareTL = squareTL;
info.description = 'Diagonal disjoint 2x2 switch gadgets';
end

function [info, corridorEdges, Ustar] = build_diagonal_corridor(edge, m)
% narrow corridor with m disjoint local switch gadgets

g = edge.g;
mask = false(edge.d, 1);
Q = zeros(edge.d, m);
squareTL = zeros(m, 2);

curR = 1;
curC = 1;

for k = 1:m
    i = 2*k - 1;
    j = 2*k - 1;
    squareTL(k,:) = [i, j];

    conn = connector_edges(curR, curC, i, j, edge.h, edge.v);
    mask(conn) = true;

    sqEdges = [
        edge.h(i,   j);
        edge.h(i+1, j);
        edge.v(i,   j);
        edge.v(i, j+1)
    ];
    mask(sqEdges) = true;

    q = zeros(edge.d, 1);
    q(edge.h(i,   j)) =  1;
    q(edge.v(i, j+1)) =  1;
    q(edge.v(i,   j)) = -1;
    q(edge.h(i+1, j)) = -1;
    Q(:,k) = q / norm(q);

    curR = i + 1;
    curC = j + 1;
end

conn = connector_edges(curR, curC, g, g, edge.h, edge.v);
mask(conn) = true;

corridorEdges = find(mask);
Ustar = Q;

info = struct();
info.squareTL = squareTL;
info.numGadgets = m;
info.description = 'Explicit diagonal corridor with local switch gadgets';
end

function E = connector_edges(r0, c0, r1, c1, h, v)
% monotone connector: R, D, R, D, ...

if r1 < r0 || c1 < c0
    error('connector_edges: endpoint must dominate startpoint.');
end

E = zeros(0,1);
r = r0;
c = c0;

while (r < r1) || (c < c1)
    if c < c1
        E(end+1,1) = h(r,c); %#ok<AGROW>
        c = c + 1;
    end
    if r < r1
        E(end+1,1) = v(r,c); %#ok<AGROW>
        r = r + 1;
    end
end
end

%==========================================================================
% PLOTTING
%==========================================================================

function make_plots(cfg, prob, edge, riskFull, riskRed, dimHist)

x = 1:cfg.nTrain;
eps0 = 1e-12;

[mF, ciF] = mean_ci90(log10(riskFull + eps0));
[mR, ciR] = mean_ci90(log10(riskRed  + eps0));

fig1 = figure('Name', sprintf('%s: test SPO risk', prob.name));
hold on; grid on; box on;
errorbar(x, mF, ciF, 'LineWidth', 1.2);
errorbar(x, mR, ciR, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+', 'Compressed SPO+ after learned subspace'}, 'Location', 'best');

switch cfg.expType
    case 'v1'
        title(sprintf('Version 1 | g=%d, d=%d, affdim(C)=d^*=%d', ...
            cfg.g, edge.d, prob.trueDstar));
    case 'v2'
        title(sprintf('Version 2 | g=%d, d=%d, true d^*=%d, affdim(C)=d', ...
            cfg.g, edge.d, prob.trueDstar));
end

mD = mean(dimHist, 1, 'omitnan');
xD = 0:cfg.nTrain;
mD0 = [0, mD];
idx = (xD <= cfg.plot.dimPlotMax);

fig2 = figure('Name', sprintf('%s: learned dimension', prob.name));
hold on; grid on; box on;
plot(xD(idx), mD0(idx), '-o', 'LineWidth', 1.6, 'MarkerSize', 4);
yline(prob.trueDstar, '--', 'LineWidth', 1.2);
xlim([0, cfg.plot.dimPlotMax]);
ylim([0, max([prob.trueDstar, mD0(idx)]) + 0.5]);
xlabel('# labeled training samples');
ylabel('dim(W) (mean over trials)');
legend({'Mean learned dim(W)', 'True d^*'}, 'Location', 'best');
title('Stage I learned dimension (early iterations only)');

if cfg.plot.saveFigures
    if exist(cfg.plot.resultsDir, 'dir') ~= 7
        mkdir(cfg.plot.resultsDir);
    end
    save_figure(fig1, fullfile(cfg.plot.resultsDir, sprintf('%s_risk.png', lower(strrep(prob.name, ' ', '_')))));
    save_figure(fig2, fullfile(cfg.plot.resultsDir, sprintf('%s_dimW.png', lower(strrep(prob.name, ' ', '_')))));
end

end

function save_figure(figHandle, filePath)
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, filePath, 'Resolution', 300);
else
    saveas(figHandle, filePath);
end
end

%==========================================================================
% STATS / UTILS
%==========================================================================

function [m, ci] = mean_ci90(M)
z = 1.645;
m = mean(M, 1, 'omitnan');
nEff = sum(~isnan(M), 1);
sd = std(M, 0, 1, 'omitnan');
se = sd ./ max(sqrt(nEff), 1);
ci = z * se;
end

function out = merge_struct(base, override)
% recursive merge for simple nested structs

out = base;
if isempty(override)
    return;
end

fn = fieldnames(override);
for k = 1:numel(fn)
    f = fn{k};
    if isstruct(override.(f)) && isfield(base, f) && isstruct(base.(f))
        out.(f) = merge_struct(base.(f), override.(f));
    else
        out.(f) = override.(f);
    end
end
end

function s = num2str_inf(x)
if isinf(x)
    s = 'inf';
else
    s = num2str(x);
end
end
