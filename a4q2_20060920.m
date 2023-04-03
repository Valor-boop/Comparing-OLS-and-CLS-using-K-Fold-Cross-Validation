function a4q2_20060920
% Starter coder for CISC371, Fall 2021, Assignment #4, Question #2

    % Acquire the instructor's data
    [xvec yvec] = clsdata;

    % Append 1's to create the design matrix
    xmat = [xvec ones(size(xvec))];

    % %
    % % STUDENT CODE GOES HERE: Set the constraint value
    % %
    theta = 8;

    % %
    % % PART (A): compare OLS and CLS on all of the data
    % %

    % Append 1's to create the design matrix
    xmat = [xvec ones(size(xvec))];

    % Compute the ordinary least squares from the normal equation
    w_ols = inv(xmat'*xmat)*xmat'*yvec;

    % Compute the constrained least squares solution
    w_cls = cls(xmat, yvec, theta);

    % PLOT: data, OLS fit, CLS fit
    plot(xvec, yvec, 'k*', xvec, polyval(w_ols, xvec), 'r-', ...
        'LineWidth', 1.5, 'MarkerSize', 8);
    %axisadjust(1.1);
    hold on;
    plot(xvec, polyval(w_cls, xvec), 'b-', ...
        'LineWidth', 1.5);
    hold off;
    tstring = strcat('\bf{}CLS fit: $\boldmath{}\|\vec{w}\|^2 = ', ...
        sprintf('%0.2f', norm(w_cls)^2), '$');
    title(tstring, 'Interpreter', 'latex', 'FontSize', 14);


    % %
    % % PART (B): compare 10 sets of 5-fold validation for OLS and CLS
    % %

    % Set the number of repetitions and number of folds
    nreps = 10;
    k = 5;

    % Set up vectors to collect results
    olsTrainVec = zeros(nreps, 1);
    olsTestVec  = zeros(nreps, 1);
    clsTrainVec = zeros(nreps, 1);
    clsTestVec  = zeros(nreps, 1);

    % Set the Random Number Generator for debugging; can comment
    % this out for results in the final report
    rng('default');

    % Run k-fold on OLS by setting theta=0
    for ix = 1:nreps
        [olsTrainVec(ix) olsTestVec(ix)] = clskfold(xmat, yvec, 0, 5);
    end

    % Re-set the RNG and run k-fold on CLS
    rng('default');
    for ix = 1:nreps
        [clsTrainVec(ix) clsTestVec(ix)] = clskfold(xmat, yvec, theta, 5);
    end
    
    % %
    % % STUDENT CODE GOES HERE: replace these 8 lines with computations
    % % of the means and standard deviations
    % %
  
    olsTrainMean   = mean(olsTrainVec);
    olsTrainStdDev = std(olsTrainVec);
    clsTrainMean   = mean(clsTrainVec);
    clsTrainStdDev = std(clsTrainVec);
    olsTestMean    = mean(olsTestVec);
    olsTestStdDev  = std(olsTestVec);
    clsTestMean    = mean(clsTestVec);
    clsTestStdDev  = std(clsTestVec);
    
    % Display the results
    disp(sprintf('   OLS results are\n     TRAIN     TEST'));
    disp([olsTrainVec olsTestVec]);
    disp(sprintf('   OLS means and std. dev. are\n    %0.4f    %0.4f\n    %0.4f    %0.4f', ...
        olsTrainMean, olsTestMean, olsTrainStdDev, olsTestStdDev));
    disp(sprintf('\n\n   CLS results are\n     TRAIN     TEST'));
    disp([clsTrainVec clsTestVec]);
    disp(sprintf('   CLS means and std. dev. are\n    %0.4f    %0.4f\n    %0.4f    %0.4f', ...
        clsTrainMean, clsTestMean, clsTrainStdDev, clsTestStdDev));

end

function [rmstrain,rmstest]=clskfold(xmat, yvec, theta, k_in)
% [RMSTRAIN,RMSTEST]=CLSKFOLD(XMAT,YVEC,THETA,K) performs a k-fold validation
% of the constrained least squares linear fit of YVEC to XMAT, with
% a solution tolerance of NORM(WCLS)^2<=THETA. See CLS for details.
% If K is omitted, the default is 5.
%
% INPUTS:
%         XMAT    - MxN data vector
%         YVEC    - Mx1 data vector
%         THETA   - positive scalar, solution threshold
%         K       - positive integer, number of folds to use
% OUTPUTS:
%         RMSTRAIN - mean root-mean-square error of training the folds
%         RMSTEST  - mean RMS error of testing the folds

% Problem size
    M = size(xmat, 1);

% Set the number of folds; must be 1<k<M
    if nargin >= 4 & ~isempty(k_in)
        k = max(min(round(k_in), M-1), 2);
    else
        k = 5;
    end

% Randomly assign the data into k folds; discard any remainders
    one2M = 1:M;
    Mk = floor(M/k);
    ndxmat = reshape(randperm(M,Mk*k), k, Mk);

% To compute RMS of fit and prediction, we will sum the variances
    vartrain  = 0.0;
    vartest = 0.0;

% Process each fold
    for ix=1:k
        ndxtrain   = reshape(ndxmat(find((1:k)~=ix), :), Mk*(k-1), 1);
        ndxtest    = reshape(ndxmat(ix, :), Mk, 1);
        xmat_train = xmat(ndxtrain,:);
        yvec_train = yvec(ndxtrain);
        xmat_test  = xmat(ndxtest,:);
        yvec_test  = yvec(ndxtest);
        w_cls = cls(xmat_train, yvec_train, theta);
        vartrain = vartrain  + rms(xmat_train*w_cls  - yvec_train )^2;
        vartest  = vartest + rms(xmat_test*w_cls - yvec_test)^2;
    end
    rmstrain = sqrt(vartrain/k);
    rmstest  = sqrt(vartest/k);
end

function [w_cls, lambda] = cls(xmat, yvec,theta)
% [WCLS,LAMBDA]=CLS(XMAT,YVEC,THETA) solves constrained
% least squares of a linear regression of YVEC to XMAT, with
% a solution tolerance of NORM(WCLS)^2<=THETA. WCLS is
% the constrained weight vector and LAMBDA is the Lagrange
% multiplier for the solution
%
% INPUTS:
%         XMAT   - MxN design matrix
%         YVEC   - Mx1 data vector
%         THETA  - positive scalar, solution threshold
% OUTPUTS:
%         WCLS   - solution coefficients
%         LAMBDA - Lagrange coefficient

% Return immediately if the threshold is invalid
    if theta<0
        w_cls = [];
        lambda = [];
        return;
    end

% Set up the problem as xmat*w=yvec
    Im = eye(size(xmat, 2));
%
% STUDENT CODE GOES HERE: define "w" and "g" functions from class notes
%
    % Equation 25.16
    wfun =@(lval) inv(xmat'*xmat + lval*Im) * xmat'*yvec;
    % Equation 25.17
    gfun =@(lval) norm(wfun(lval))^2 - theta;

% OLS solution: use pseudo-inverse for ill conditioned matrix
    if cond(xmat)<1e+8
        wls = xmat\yvec;
    else
        wls = pinv(xmat)*vec;
    end

% The OLS solution is used if it is within the user's threshold
    if norm(wls)^2<= theta | theta<=0
        w_cls = wls;
        lambda = 0;
    else
% %
% % STUDENT CODE GOES HERE: replace these 2 lines
% % You can use "fzero" to estimate lambda
% %
        lambda = fzero(gfun, 0);
        w_cls = wfun(lambda);
    end
end

function [xvec, yvec] = clsdata
% [XVEC,YVEC]=CLSDATA creates a small data set for testing
% linear regression. XVEC contains equally spaced points.
% YVEC is mainly an affine transformation of XVEC, with the first
% and last values deviated.
%
% INPUTS:
%         none
% OUTPUTS:
%         XVEC - Mx1 vector of independent data
%         YVEC - Mx1 vector of   dependent data

% X values are equally spaced
    xvec = linspace(0, 9, 10)';

% Y are linear, deviating first and last
    ylin = exp(1)*xvec + pi;
    yvec = [(ylin(1) - 5) ; ylin(2:end-1) ; ylin(end) + 3];
end
