
function Version_3_structured_full_dimension_20x20()
%==========================================================================
% Version 3: structured full-dimensional 20x20 instance
%
% Goal:
%   Make Stage I closer to the paper's Algorithm 4 + Algorithm 2.
%
% What changes relative to Version 2:
%   1) Stage I still fits centered OLS for the conditional mean.
%   2) It then forms C-valued pseudo-costs on fresh discovery contexts.
%   3) Instead of SVD on pseudo-costs, it runs a cumulative warm-started
%      pointwise routine over pseudo-costs.
%   4) For each pseudo-cost c_hat, the pointwise routine:
%        - solves x*(c_hat),
%        - enumerates adjacent corridor paths obtained by flipping one
%          gadget,
%        - checks whether the current fiber still allows the adjacent path
%          to beat x*(c_hat),
%        - if yes, adds the path-difference direction x*(c_hat)-x' to the
%          dataset,
%        - repeats until pointwise sufficiency holds for that pseudo-cost.
%
% This file is specialized to the planted corridor construction used in
% Version 2. It is not a generic implementation of Algorithm 1/2 for
% arbitrary LPs, but it is much closer to the paper's Stage-I logic than
% the current pseudo-cost SVD baseline.
%
% Explicit true subspace in this instance:
%   W* = span{q_1,...,q_m},
% where each q_k is the local path-difference vector on the k-th 2x2
% corridor gadget:
%   q_k = (RD on gadget k) - (DR on gadget k).
%
% In particular, if x^0 is the "all-DR" corridor path and sigma in {0,1}^m,
% then every corridor path can be written as
%   x^sigma = x^0 + sum_k sigma_k q_k.
% Because any corridor choice pattern can be made optimal by a suitable
% cost in the box prior, we have
%   X*(C) = {x^sigma : sigma in {0,1}^m},
%   W*    = dir(X*(C)) = span{q_1,...,q_m}.
%==========================================================================

clc;
close all;

% ----------------------------- Configuration -----------------------------
cfg.seed         = 202603131;
cfg.g            = 20;      % 20x20 node grid
cfg.p            = 50;      % context dimension
cfg.dstar_target = 8;       % planted / true decision-relevant dimension

cfg.trainSizes   = [80 160 320 640 960];
cfg.nTest        = 1000;
cfg.nTrial       = 5;

% Full-dimensional box prior
cfg.lowBase      = 10;
cfg.radCorr      = 1;       % corridor edges in [9,11]
cfg.highBase     = 100;
cfg.radOut       = 1;       % outside edges in [99,101]

% Contextual signal and noise
cfg.signalAmp    = 1.00;
cfg.noiseCorr    = 0.05;
cfg.noiseOut     = 0.02;

% Stage I (Algorithm-4-like)
cfg.rankTol      = 5e-2;    % kept only for optional diagnostics
cfg.maxRank      = 12;      % kept only for optional diagnostics
cfg.ridge        = 1e-6;
cfg.stage1Tol    = 1e-10;
cfg.indepTol     = 1e-10;

% Stage II (SPO+)
cfg.numEpochs    = 12;
cfg.lrFull       = 0.010;
cfg.lrRed        = 0.030;
cfg.batchSize    = 64;
cfg.gradClip     = 5.0;

rng(cfg.seed, 'twister');

% ----------------------------- Build problem -----------------------------
edge = build_grid_edge_maps(cfg.g);
d    = edge.d;
L    = 2 * (cfg.g - 1);

[problem, cfg] = build_structured_problem(edge, cfg);

fprintf('=== Version 3, Algorithm-4/2-like Stage I on structured 20x20 instance ===\n');
fprintf('Grid size g = %d, #edges d = %d, path length L = %d\n', cfg.g, d, L);
fprintf('affdim(C) = %d (should equal d = %d)\n', problem.affdimC, d);
fprintf('True decision-relevant dimension d_* = %d\n', problem.rstar);
fprintf('Corridor edges = %d / %d\n', numel(problem.corridorEdges), d);
verify_domination_sufficient_condition(cfg);

nN = numel(cfg.trainSizes);

riskFull   = zeros(cfg.nTrial, nN);
riskLearn  = zeros(cfg.nTrial, nN);
riskOracle = zeros(cfg.nTrial, nN);

dimW        = zeros(cfg.nTrial, nN);
captureTrue = zeros(cfg.nTrial, nN);
captureHat  = zeros(cfg.nTrial, nN);
maxAngleDeg = zeros(cfg.nTrial, nN);
nHard       = zeros(cfg.nTrial, nN);

% -------------------------------- Trials ---------------------------------
for ii = 1:nN
    nTrain = cfg.trainSizes(ii);
    fprintf('\n--- nTrain = %d ---\n', nTrain);

    for tr = 1:cfg.nTrial
        rng(cfg.seed + 100 * ii + tr, 'twister');

        Atrue = randn(problem.rstar, cfg.p);

        [Xtr, Ctr] = sample_corridor_costs( ...
            nTrain, cfg.p, problem.cBase, problem.Utrue, Atrue, ...
            cfg.signalAmp, cfg.noiseCorr, cfg.noiseOut, ...
            problem.lbC, problem.ubC, problem.corridorEdges);

        [Xte, Cte] = sample_corridor_costs( ...
            cfg.nTest, cfg.p, problem.cBase, problem.Utrue, Atrue, ...
            cfg.signalAmp, cfg.noiseCorr, cfg.noiseOut, ...
            problem.lbC, problem.ubC, problem.corridorEdges);

        % ---------------- Stage I: Algorithm-4-like ---------------------
        nReg  = max(cfg.p + 5, floor(0.5 * nTrain));
        nDisc = max(cfg.p + 5, nTrain - nReg);

        Xreg = Xtr(1:nReg, :);
        Creg = Ctr(1:nReg, :);

        [stage1, Ulearn] = stage1_algorithm4_like( ...
            Xreg, Creg, nDisc, problem, cfg);

        subStats = compare_subspaces(Ulearn, problem.Utrue);

        % ---------------- Stage II: full / learned / oracle -------------
        Bfull = train_spoplus_full_dp( ...
            Xtr, Ctr, edge, problem.cBase, ...
            cfg.numEpochs, cfg.lrFull, cfg.batchSize, cfg.gradClip, ...
            problem.lbC, problem.ubC);

        Glearn = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, problem.cBase, Ulearn, ...
            cfg.numEpochs, cfg.lrRed, cfg.batchSize, cfg.gradClip, ...
            problem.lbC, problem.ubC);

        Goracle = train_spoplus_reduced_dp( ...
            Xtr, Ctr, edge, problem.cBase, problem.Utrue, ...
            cfg.numEpochs, cfg.lrRed, cfg.batchSize, cfg.gradClip, ...
            problem.lbC, problem.ubC);

        % ---------------- Evaluation ------------------------------------
        riskFull(tr, ii) = mean_spo_risk_full( ...
            Bfull, Xte, Cte, edge, problem.cBase, problem.lbC, problem.ubC);

        riskLearn(tr, ii) = mean_spo_risk_reduced( ...
            Glearn, Ulearn, Xte, Cte, edge, problem.cBase, ...
            problem.lbC, problem.ubC);

        riskOracle(tr, ii) = mean_spo_risk_reduced( ...
            Goracle, problem.Utrue, Xte, Cte, edge, problem.cBase, ...
            problem.lbC, problem.ubC);

        dimW(tr, ii)        = size(Ulearn, 2);
        captureTrue(tr, ii) = subStats.captureTrueInHat;
        captureHat(tr, ii)  = subStats.captureHatInTrue;
        maxAngleDeg(tr, ii) = subStats.maxPrincipalAngleDeg;
        nHard(tr, ii)       = stage1.nHard;

        fprintf(['trial %2d/%2d | dim(W)=%.0f | hard=%d | ' ...
                 'capture=%.3f | maxAngle=%.2f deg | ' ...
                 'risk_full=%.4g | risk_learn=%.4g | risk_oracle=%.4g\n'], ...
                 tr, cfg.nTrial, dimW(tr,ii), nHard(tr,ii), ...
                 captureTrue(tr,ii), maxAngleDeg(tr,ii), ...
                 riskFull(tr,ii), riskLearn(tr,ii), riskOracle(tr,ii));
    end
end

% ----------------------------- Summaries ---------------------------------
xAxis = cfg.trainSizes;

[mF, ciF] = mean_ci90(log10(riskFull   + 1e-12));
[mL, ciL] = mean_ci90(log10(riskLearn  + 1e-12));
[mO, ciO] = mean_ci90(log10(riskOracle + 1e-12));

mD   = mean(dimW, 1, 'omitnan');
mCap = mean(captureTrue, 1, 'omitnan');
mAng = mean(maxAngleDeg, 1, 'omitnan');
mH   = mean(nHard, 1, 'omitnan');

paramFull  = d * (cfg.p + 1);
paramLearn = mean(dimW(:,end)) * (cfg.p + 1);
paramOrac  = problem.rstar * (cfg.p + 1);

fprintf('\n=== Version 3 summary ===\n');
fprintf('Full linear model params          = d*(p+1) = %d\n', paramFull);
fprintf('Oracle reduced model params       = d_*(p+1) = %d\n', paramOrac);
fprintf('Mean final learned dim(W)         = %.2f\n', mean(dimW(:,end)));
fprintf('Approx learned reduced params     = %.2f\n', paramLearn);
fprintf('Mean final capture(W* in W_hat)   = %.4f\n', mean(captureTrue(:,end)));
fprintf('Mean final max principal angle    = %.4f deg\n', mean(maxAngleDeg(:,end)));
fprintf('Mean final #hard pseudo-costs     = %.2f\n', mean(nHard(:,end)));

% -------------------------------- Plots ----------------------------------
figRisk = figure('Name', 'Version 3 (20x20): Test SPO risk');
hold on; grid on; box on;
errorbar(xAxis, mF, ciF, 'LineWidth', 1.2);
errorbar(xAxis, mL, ciL, 'LineWidth', 1.2);
errorbar(xAxis, mO, ciO, 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('log10(Test SPO risk)');
legend({'Full SPO+', 'Reduced SPO+ (learned W)', 'Reduced SPO+ (oracle W*)'}, ...
       'Location', 'best');
title(sprintf('Version 3, g=%d, d=%d, true d_*=%d, affdim(C)=d', ...
      cfg.g, d, problem.rstar));

figStage = figure('Name', 'Version 3 (20x20): Stage-I diagnostics');
subplot(2, 2, 1);
hold on; grid on; box on;
plot([0, xAxis], [0, mD], '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
yline(problem.rstar, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('dim(W)');
legend({'Mean learned dim(W)', 'True d_*'}, 'Location', 'best');
title('Dimension of learned subspace');

subplot(2, 2, 2);
hold on; grid on; box on;
plot(xAxis, mCap, '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
yline(1.0, '--', 'LineWidth', 1.2);
xlabel('# labeled training samples');
ylabel('capture(W* in W_hat)');
title('Subspace capture');

subplot(2, 2, 3);
hold on; grid on; box on;
plot(xAxis, mAng, '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
xlabel('# labeled training samples');
ylabel('max principal angle (deg)');
title('Largest principal angle to W*');

subplot(2, 2, 4);
hold on; grid on; box on;
plot(xAxis, mH, '-o', 'LineWidth', 1.6, 'MarkerSize', 5);
xlabel('# labeled training samples');
ylabel('mean # hard pseudo-costs');
title('Algorithm-2-like hard set size');

resultsDir = prepare_results_dir();
save_figure(figRisk,  fullfile(resultsDir, 'version3_structured_20x20_spo_risk.png'));
save_figure(figStage, fullfile(resultsDir, 'version3_structured_20x20_stage1_diag.png'));

save(fullfile(resultsDir, 'version3_structured_20x20_summary.mat'), ...
    'cfg', 'problem', ...
    'riskFull', 'riskLearn', 'riskOracle', ...
    'dimW', 'captureTrue', 'captureHat', 'maxAngleDeg', 'nHard', ...
    'mF', 'ciF', 'mL', 'ciL', 'mO', 'ciO', 'mD', 'mCap', 'mAng', 'mH', ...
    'paramFull', 'paramLearn', 'paramOrac');

fprintf('Saved results to %s\n', resultsDir);
end

%==========================================================================
% Problem builder
%==========================================================================
function [problem, cfg] = build_structured_problem(edge, cfg)
[gadgetInfo, corridorEdges, Utrue, QtrueRaw] = build_diagonal_corridor_detailed(edge, cfg.dstar_target);

rstar = size(Utrue, 2);
d     = edge.d;

lbC = (cfg.highBase - cfg.radOut) * ones(d,1);
ubC = (cfg.highBase + cfg.radOut) * ones(d,1);

lbC(corridorEdges) = cfg.lowBase - cfg.radCorr;
ubC(corridorEdges) = cfg.lowBase + cfg.radCorr;

cBase   = 0.5 * (lbC + ubC);
affdimC = sum(ubC > lbC + 1e-12);

problem.edge          = edge;
problem.gadgetInfo    = gadgetInfo;
problem.corridorEdges = corridorEdges;
problem.lbC           = lbC;
problem.ubC           = ubC;
problem.cBase         = cBase;
problem.affdimC       = affdimC;
problem.Utrue         = Utrue;
problem.QtrueRaw      = QtrueRaw;
problem.rstar         = rstar;
end

%==========================================================================
% Stage I: Algorithm-4-like OLS + pseudo-costs + Algorithm-2-like routine
%==========================================================================
function [stage1, Ulearn] = stage1_algorithm4_like(Xreg, Creg, nDisc, problem, cfg)
p = size(Xreg, 2);
d = size(Creg, 2);

% Step 1 of Algorithm 4: centered OLS for mu(x)-cBase
if size(Xreg, 1) < p
    Ahat = zeros(d, p);
else
    Yreg = Creg - problem.cBase.';
    XtX  = Xreg.' * Xreg + cfg.ridge * eye(p);
    Ahat = (Yreg.' * Xreg) / XtX;  % d x p
end

% Step 2 of Algorithm 4: fresh contexts -> pseudo-costs, projected to C
Xdisc    = randn(nDisc, p);
ChatDisc = problem.cBase.' + Xdisc * Ahat.';
ChatDisc = min(max(ChatDisc, problem.lbC.'), problem.ubC.');

% Optional SVD baseline on pseudo-costs (for debugging only)
Usvd = learn_basis_from_centered_samples(ChatDisc - problem.cBase.', cfg.rankTol, cfg.maxRank);

% Step 3 of Algorithm 4: run Algorithm 2-like cumulative pointwise routine
D = zeros(problem.edge.d, 0);
learnedGadgets = false(problem.rstar, 1);

hardMask      = false(nDisc, 1);
perCostAdds   = zeros(nDisc, 1);
dimAfterCost  = zeros(nDisc, 1);

for i = 1:nDisc
    cIn = ChatDisc(i, :).';

    [Dnew, learnedGadgetsNew, addCount] = pointwise_routine_corridor( ...
        cIn, D, learnedGadgets, problem, cfg);

    if addCount > 0
        hardMask(i) = true;
    end

    D              = Dnew;
    learnedGadgets = learnedGadgetsNew;
    perCostAdds(i) = addCount;
    dimAfterCost(i)= size(D, 2);

    if all(learnedGadgets)
        % Continuing would not change anything in this planted instance.
        break;
    end
end

Ulearn = D;

stage1.Ahat            = Ahat;
stage1.Xdisc           = Xdisc;
stage1.ChatDisc        = ChatDisc;
stage1.D               = D;
stage1.Usvd            = Usvd;
stage1.hardMask        = hardMask;
stage1.hardSet         = find(hardMask);
stage1.nHard           = nnz(hardMask);
stage1.perCostAdds     = perCostAdds;
stage1.dimAfterCost    = dimAfterCost;
stage1.learnedGadgets  = learnedGadgets;
end

%==========================================================================
% Pointwise routine specialized to the Version-2 corridor construction
%==========================================================================
function [Dout, learnedGadgets, addCount] = pointwise_routine_corridor(cIn, Dinit, learnedInit, problem, cfg)
Dout          = Dinit;
learnedGadgets = learnedInit;
addCount      = 0;

while true
    % Anchor at cIn, as in Algorithm 1
    [~, xAnchor] = oracle_monotone_path_dp(cIn, problem.edge);
    gadgetChoice = infer_gadget_choices_from_path(xAnchor, problem.gadgetInfo);

    foundViolation = false;
    bestScore      = -inf;
    bestK          = NaN;
    bestQ          = [];
    bestWitness    = [];

    % Enumerate adjacent corridor paths: flip exactly one gadget
    for k = 1:problem.rstar
        xAlt = flip_single_gadget(xAnchor, gadgetChoice, problem.gadgetInfo, k);
        q    = full(xAnchor - xAlt);  % user-requested orientation x*(c)-x'

        [isViol, witnessScore, cWitness] = ...
            fiber_admits_adjacent_better(q, cIn, learnedGadgets(k), Dout, problem, cfg);

        if isViol
            foundViolation = true;
            if witnessScore > bestScore
                bestScore   = witnessScore;
                bestK       = k;
                bestQ       = q;
                bestWitness = cWitness;
            end
        end
    end

    if ~foundViolation
        break;
    end

    % Add one violated facet/path-difference at a time (Algorithm-1-like)
    qNew = bestQ;

    if learnedGadgets(bestK)
        break;
    end

    Dout = append_orth_direction(Dout, qNew, cfg.indepTol);
    learnedGadgets(bestK) = true;
    addCount = addCount + 1;

    if size(Dout, 2) >= problem.rstar
        break;
    end
end
end

function [isViol, witnessScore, cWitness] = fiber_admits_adjacent_better(q, cIn, gadgetAlreadyLearned, Dcur, problem, cfg)
% In this planted box-prior corridor instance, if the gadget underlying q
% has not yet been queried, then the current fiber still leaves that local
% cost difference unconstrained. We can explicitly construct a witness cost
% in the fiber that makes the adjacent path cheaper than the anchor path.

if gadgetAlreadyLearned
    cWitness     = cIn;
    witnessScore = q.' * cWitness;
    isViol       = false;
    return;
end

cWitness = cIn;

posIdx = find(q >  0.5);  % anchor-path edges on the flipped gadget
negIdx = find(q < -0.5);  % adjacent-path edges on the flipped gadget

cWitness(posIdx) = problem.ubC(posIdx);
cWitness(negIdx) = problem.lbC(negIdx);

% Since supports of learned gadget directions are disjoint, all existing
% measurements remain unchanged.
if ~isempty(Dcur)
    resid = norm(Dcur.' * (cWitness - cIn), inf);
    if resid > 1e-8
        error('fiber_admits_adjacent_better: witness violates current measurements.');
    end
end

% Adjacent path is cheaper than anchor path iff q^T cWitness > 0
witnessScore = q.' * cWitness;
isViol       = (witnessScore > cfg.stage1Tol);
end

%==========================================================================
% Subspace diagnostics
%==========================================================================
function stats = compare_subspaces(Uhat, Utrue)
Uhat  = orth(Uhat);
Utrue = orth(Utrue);

stats.dimHat  = size(Uhat, 2);
stats.dimTrue = size(Utrue, 2);

if isempty(Uhat) || isempty(Utrue)
    stats.captureTrueInHat    = 0;
    stats.captureHatInTrue    = 0;
    stats.maxPrincipalAngleDeg= 90;
    stats.minPrincipalAngleDeg= 90;
    return;
end

sv = svd(Uhat.' * Utrue);
sv = max(min(sv, 1), -1);
ang = acosd(sv);

if size(Uhat,2) ~= size(Utrue,2)
    nMiss = abs(size(Uhat,2) - size(Utrue,2));
    ang   = [ang; 90 * ones(nMiss, 1)];
end

stats.captureTrueInHat     = norm(Uhat.' * Utrue, 'fro')^2 / size(Utrue, 2);
stats.captureHatInTrue     = norm(Utrue.' * Uhat, 'fro')^2 / size(Uhat, 2);
stats.maxPrincipalAngleDeg = max(ang);
stats.minPrincipalAngleDeg = min(ang);
end

function U = append_orth_direction(U, q, tol)
if isempty(q)
    return;
end

if norm(q, 2) < tol
    return;
end

if isempty(U)
    U = q / norm(q, 2);
    return;
end

qPerp = q - U * (U.' * q);
nrm   = norm(qPerp, 2);

if nrm > tol
    U = [U, qPerp / nrm];
end
end

%==========================================================================
% Data generation
%==========================================================================
function [X, C] = sample_corridor_costs(n, p, cBase, Ustar, Atrue, signalAmp, noiseCorr, noiseOut, lbC, ubC, corridorEdges)
% Full-dimensional box prior:
% every edge varies in an interval => affdim(C)=d
%
% Decision-relevant contextual signal:
% only injected along the true low-dimensional corridor-switch basis Ustar

d = numel(cBase);

X = randn(n, p);
Lat = tanh((Atrue * X.') / sqrt(p));   % r x n

Signal = signalAmp * (Ustar * Lat);    % d x n

Noise = noiseOut * (2 * rand(d, n) - 1);
Noise(corridorEdges, :) = noiseCorr * (2 * rand(numel(corridorEdges), n) - 1);

Cmat = repmat(cBase, 1, n) + Signal + Noise;
Cmat = min(max(Cmat, lbC), ubC);

C = Cmat.';
end

%==========================================================================
% Optional SVD helper (for debugging only)
%==========================================================================
function U = learn_basis_from_centered_samples(M, tolRel, maxRank)
if isempty(M)
    U = zeros(size(M,2), 0);
    return;
end

[Utmp, S, ~] = svd(M.', 'econ');
s = diag(S);

if isempty(s) || s(1) < 1e-12
    U = zeros(size(M,2), 0);
    return;
end

r = sum((s >= tolRel * s(1)) & (s > 1e-10));
r = min(r, maxRank);

if r == 0
    U = zeros(size(M,2), 0);
else
    U = Utmp(:, 1:r);
end
end

%==========================================================================
% Stage II: SPO+ training
%==========================================================================
function B = train_spoplus_full_dp(X, C, edge, cRef, numEpochs, lr0, batchSize, gradClip, lbPred, ubPred)
[n, p] = size(X);
d = edge.d;
B = zeros(d, p + 1);

for ep = 1:numEpochs
    eta  = lr0 / sqrt(ep);
    perm = randperm(n);

    for startIdx = 1:batchSize:n
        ids  = perm(startIdx:min(startIdx + batchSize - 1, n));
        Grad = zeros(d, p + 1);

        for kk = 1:numel(ids)
            i   = ids(kk);
            phi = [X(i,:)'; 1];
            c   = C(i,:)';

            chat = cRef + B * phi;
            chat = min(max(chat, lbPred), ubPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + full(subg) * phi.';
        end

        Grad  = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        B = B - eta * Grad;
    end
end
end

function G = train_spoplus_reduced_dp(X, C, edge, cRef, U, numEpochs, lr0, batchSize, gradClip, lbPred, ubPred)
[n, p] = size(X);
r = size(U, 2);

if r == 0
    G = zeros(0, p + 1);
    return;
end

G = zeros(r, p + 1);

for ep = 1:numEpochs
    eta  = lr0 / sqrt(ep);
    perm = randperm(n);

    for startIdx = 1:batchSize:n
        ids  = perm(startIdx:min(startIdx + batchSize - 1, n));
        Grad = zeros(r, p + 1);

        for kk = 1:numel(ids)
            i   = ids(kk);
            phi = [X(i,:)'; 1];
            c   = C(i,:)';

            chat = cRef + U * (G * phi);
            chat = min(max(chat, lbPred), ubPred);

            subg = spoplus_subgrad_dp(chat, c, edge);
            Grad = Grad + (U.' * full(subg)) * phi.';
        end

        Grad  = Grad / numel(ids);
        gnorm = norm(Grad(:), 2);
        if gnorm > gradClip
            Grad = Grad * (gradClip / gnorm);
        end

        G = G - eta * Grad;
    end
end
end

%==========================================================================
% Evaluation
%==========================================================================
function risk = mean_spo_risk_full(B, X, C, edge, cRef, lbPred, ubPred)
n = size(X, 1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c   = C(i,:)';

    chat = cRef + B * phi;
    chat = min(max(chat, lbPred), ubPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

function risk = mean_spo_risk_reduced(G, U, X, C, edge, cRef, lbPred, ubPred)
n = size(X, 1);
tot = 0;

for i = 1:n
    phi = [X(i,:)'; 1];
    c   = C(i,:)';

    if isempty(U)
        chat = cRef;
    else
        chat = cRef + U * (G * phi);
    end
    chat = min(max(chat, lbPred), ubPred);

    tot = tot + spo_loss_dp(chat, c, edge);
end

risk = tot / n;
end

%==========================================================================
% SPO / SPO+ primitives
%==========================================================================
function loss = spo_loss_dp(chat, ctrue, edge)
[~, wHat] = oracle_monotone_path_dp(chat, edge);
[~, wOpt] = oracle_monotone_path_dp(ctrue, edge);
loss = ctrue.' * (wHat - wOpt);
end

function subg = spoplus_subgrad_dp(chat, ctrue, edge)
[~, w0] = oracle_monotone_path_dp(ctrue, edge);
[~, w1] = oracle_monotone_path_dp(2 * chat - ctrue, edge);
subg = 2 * (w0 - w1);
end

%==========================================================================
% Monotone shortest-path oracle by dynamic programming
%==========================================================================
function [bestCost, w] = oracle_monotone_path_dp(c, edge)
g = edge.g;
h = edge.h;
v = edge.v;

D      = inf(g, g);
parent = zeros(g, g, 'uint8');   % 1 = from left, 2 = from up

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
                parent(i, j+1) = 1;
            end
        end

        if i < g
            cand = cur + c(v(i,j));
            if cand < D(i+1, j)
                D(i+1, j) = cand;
                parent(i+1, j) = 2;
            end
        end
    end
end

bestCost = D(g, g);
w = sparse(edge.d, 1);

i = g;
j = g;
while (i > 1) || (j > 1)
    if parent(i,j) == 1
        e = h(i, j-1);
        w(e) = 1;
        j = j - 1;
    elseif parent(i,j) == 2
        e = v(i-1, j);
        w(e) = 1;
        i = i - 1;
    else
        error('oracle_monotone_path_dp: failed to reconstruct path.');
    end
end
end

%==========================================================================
% Corridor construction and explicit W*
%==========================================================================
function [info, corridorEdges, Utrue, Qraw] = build_diagonal_corridor_detailed(edge, m)
% Construct a narrow corridor with exactly m disjoint local switch gadgets.
%
% All path differences among corridor paths live in span{q_1,...,q_m}.
% Each q_k is the local difference:
%   q_k = incidence(RD on gadget k) - incidence(DR on gadget k).

g = edge.g;
maxPossible = floor((g - 1) / 2);
if m > maxPossible
    error('Requested m=%d gadgets, but at most %d fit on a %dx%d grid.', ...
          m, maxPossible, g, g);
end

mask      = false(edge.d, 1);
Qraw      = zeros(edge.d, m);
squareTL  = zeros(m, 2);
gadgets   = repmat(struct(), m, 1);

curR = 1;
curC = 1;

for k = 1:m
    i = 2 * k - 1;
    j = 2 * k - 1;

    squareTL(k,:) = [i, j];

    conn = connector_edges(curR, curC, i, j, edge.h, edge.v);
    mask(conn) = true;

    topH   = edge.h(i,   j);
    botH   = edge.h(i+1, j);
    leftV  = edge.v(i,   j);
    rightV = edge.v(i,   j+1);

    RD = [topH; rightV];
    DR = [leftV; botH];

    sqEdges = [topH; botH; leftV; rightV];
    mask(sqEdges) = true;

    q = zeros(edge.d, 1);
    q(topH)   =  1;
    q(rightV) =  1;
    q(leftV)  = -1;
    q(botH)   = -1;
    Qraw(:,k) = q;

    gadgets(k).topH    = topH;
    gadgets(k).bottomH = botH;
    gadgets(k).leftV   = leftV;
    gadgets(k).rightV  = rightV;
    gadgets(k).RD      = RD;
    gadgets(k).DR      = DR;
    gadgets(k).support = sqEdges;
    gadgets(k).squareTL= [i, j];

    curR = i + 1;
    curC = j + 1;
end

conn = connector_edges(curR, curC, g, g, edge.h, edge.v);
mask(conn) = true;

corridorEdges = find(mask);
Utrue = orth(Qraw);

info.squareTL   = squareTL;
info.numGadgets = m;
info.gadgets    = gadgets;
end

function E = connector_edges(r0, c0, r1, c1, h, v)
% Deterministic monotone connector: R, D, R, D, ...
if r1 < r0 || c1 < c0
    error('connector_edges: end point must dominate start point.');
end

E = zeros(0,1);
r = r0;
c = c0;

while (r < r1) || (c < c1)
    if c < c1
        E(end+1,1) = h(r, c); %#ok<AGROW>
        c = c + 1;
    end
    if r < r1
        E(end+1,1) = v(r, c); %#ok<AGROW>
        r = r + 1;
    end
end
end

function verify_domination_sufficient_condition(cfg)
L = 2 * (cfg.g - 1);

% worst corridor path cost
maxCorr = L * (cfg.lowBase + cfg.radCorr);

% any non-corridor path must use at least one outside edge
minOneOutside = (cfg.highBase - cfg.radOut) + ...
                (L - 1) * (cfg.lowBase - cfg.radCorr);

fprintf('Domination sufficient condition:\n');
fprintf(' max corridor-path cost <= %.2f\n', maxCorr);
fprintf(' min path cost with one outside edge >= %.2f\n', minOneOutside);

if minOneOutside <= maxCorr
    warning('Outside-edge penalty is not large enough. Increase highBase.');
else
    fprintf(' OK: any path using an outside edge is always worse than a pure corridor path.\n');
end
end

%==========================================================================
% Path / gadget helpers
%==========================================================================
function gadgetChoice = infer_gadget_choices_from_path(x, info)
m = info.numGadgets;
gadgetChoice = zeros(m, 1);  % +1 = RD, -1 = DR

for k = 1:m
    g = info.gadgets(k);

    useRD = (full(x(g.RD(1))) > 0.5) && (full(x(g.RD(2))) > 0.5);
    useDR = (full(x(g.DR(1))) > 0.5) && (full(x(g.DR(2))) > 0.5);

    if useRD && ~useDR
        gadgetChoice(k) = +1;
    elseif useDR && ~useRD
        gadgetChoice(k) = -1;
    else
        error('infer_gadget_choices_from_path: path does not use exactly one local route on gadget %d.', k);
    end
end
end

function xAlt = flip_single_gadget(xAnchor, gadgetChoice, info, k)
xAlt = xAnchor;
g = info.gadgets(k);

if gadgetChoice(k) == +1
    % current path uses RD, flip to DR
    xAlt(g.RD) = 0;
    xAlt(g.DR) = 1;
elseif gadgetChoice(k) == -1
    % current path uses DR, flip to RD
    xAlt(g.DR) = 0;
    xAlt(g.RD) = 1;
else
    error('flip_single_gadget: invalid gadget choice.');
end
end

%==========================================================================
% Geometry: edge maps
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

edge.h = h;
edge.v = v;
edge.g = g;
edge.d = idx - 1;
end

%==========================================================================
% Helpers
%==========================================================================
function [m, ci] = mean_ci90(M)
z = 1.645;
m = mean(M, 1, 'omitnan');
nEff = sum(~isnan(M), 1);
sd = std(M, 0, 1, 'omitnan');
se = sd ./ max(sqrt(nEff), 1);
ci = z * se;
end

function resultsDir = prepare_results_dir()
repoDir = fileparts(mfilename('fullpath'));
resultsDir = fullfile(repoDir, 'results');
if exist(resultsDir, 'dir') ~= 7
    mkdir(resultsDir);
end
end

function save_figure(figHandle, filePath)
if exist('exportgraphics', 'file') == 2
    exportgraphics(figHandle, filePath, 'Resolution', 300);
else
    saveas(figHandle, filePath);
end
end
