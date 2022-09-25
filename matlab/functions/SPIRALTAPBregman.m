function [x] = SPIRALTAPBregman(y, A,Aprime,xinit, tau, miniter,maxiter)
%===============================================================================
%This is SPIRAL-TAP:                                                         
%      Sparse Poisson Intensity Reconstruction Algorithms -- Theory and Practice
%===============================================================================
%SPIRALTAPCanonical - Version 2.0
%   Copyright 2009, 2010, 2011                                                      
%   Zachary T. Harmany*, Roummel F. Marcia**, Rebecca M. Willett*             
%       *  Department of Electrical and Computer Engineering                  
%          Duke University                                                    
%          Durham, NC 27708, USA                                              
%      **  School of Natural Sciences                                         
%          University of California, Merced                                   
%          Merced, CA 95343, USA                                              
%   Corresponding author: Zachary T. Harmany (zth@duke.edu)                   
%=============================================================================== 
%[x, varargout] = SPIRALTAPCanonical(y, A, tau, varargin)
%
%This function solves the optimization problem
%
%   minimize - log p( y | A x ) + tau ||x||_1                           
%      x
%   subject to x >= 0                                                    
%
%where p( y | A x ) is the Poisson likelihood, tau a regularization parameter.
%If tau is a vector (of the same size as x), then this function will also solve 
%                                                                        
%   minimize - log p( y | A x ) + || diag(tau) x||_1 
%      x
%   subject to x >= 0                                                    
%
%where diag(tau) is a diagonal matrix with entries specified by tau.     
%===============================================================================
%Required inputs:
%   y
%   A
%   tau
%
%===============================================================================
%Development Notes:
% - encorporate checks to the input parameters
% - warnings expert user flags
% - write documentation and help
%===============================================================================

%====================================
%= Set default / initial parameters =
%====================================
% Problem specification options
%AT = [];
%simplebound = 1;
%lowerbound = 0;
truth = [];
%init = 0;
% Treatment of alpha (step size) parameter
alpha = 1;
% alphamin = 1e-30;
% alphamax = 1e+30;

alphamin = 1e-30;
alphamax = 1e+30;


%alphamethod = 1;
monotone = 1;
acceptalphamax = 1e+30;
acceptmult = 2;
acceptdecrease = 0.1;
acceptpast = 10;
% Algorithm termination options
%stopcriterion = 3;
tolerance = (1e-6)^2;%%modification no more sqrt

% Output options
verbose = 0;
saveobjective = 0;
savereconerror = 0;
savecputime = 0;
reconerrortype = 0;

%saveiteratepath = 0; % Can be memory intensive, don't save by default
%savesteppath = 0; 
% Advanced options
%warnings = 1;
% Set non-optional initial parameters
converged = 0;
iter = 0;
% Reserve variable names
termval = [];
objective = [];
reconerror = [];
cputime = [];
alphapath = [];
% iteratepath = [];
% steppath = [];

% switch nargout
%   case 4; saveobjective = 1; savealphapath = 1;
% 
%   case 5; saveobjective = 1; savereconerror = 1;
%   case 6; saveobjective = 1; savereconerror = 1; savecputime = 1;
%   case 7; saveobjective = 1; savereconerror = 1; savecputime = 1; 
%           savealphapath = 1;
%   case 8; saveobjective = 1; savereconerror = 1; savecputime = 1; 
%           savealphapath = 1; saveiteratepath = 1;
%   case 9; saveobjective = 1; savereconerror = 1; savecputime = 1; 
%           savealphapath = 1; saveiteratepath = 1; savesteppath = 1;
% end
    


%========================================
%= Read in options and input parameters =
%========================================
% if (rem(length(varargin),2)==1)
% 	error('Optional parameters should always go by pairs.');
% else
%   for ii = 1:2:(length(varargin)-1)
%     switch lower(varargin{ii})
%       % Problem specification options
% %       case 'at';                  AT                  = varargin{ii+1};
% %       case 'lowerbound';          lowerbound          = varargin{ii+1};
% %       case 'truth';               truth               = varargin{ii+1};
% %       case 'init';                init                = varargin{ii+1};
%       % Treatment of alpha (step size) parameter
% %       case 'alpha';               alpha               = varargin{ii+1};
% %       case 'alphamin';            alphamin            = varargin{ii+1};
% %       case 'alphamax';            alphamax            = varargin{ii+1};
% %       case 'alphamethod';         alphamethod         = varargin{ii+1};
% %      case 'monotone';            monotone            = varargin{ii+1};
% %      case 'acceptalphamax';      acceptalphamax      = varargin{ii+1};
% %       case 'eta';                 acceptmult          = varargin{ii+1};
% %       case 'sigma';               acceptdecrease      = varargin{ii+1};
% %       case 'acceptpast';          acceptpast          = varargin{ii+1};
%       % Algorithm termination options
% %       case 'miniter';             miniter             = varargin{ii+1}; 
% %       case 'maxiter';             maxiter             = varargin{ii+1}; 
% %       case 'stopcriterion';       stopcriterion       = varargin{ii+1};   
% %       case 'tolerance';           tolerance           = varargin{ii+1}; 
%       % Output options
% %       case 'verbose';             verbose             = varargin{ii+1};
% %       case 'saveobjective';       saveobjective       = varargin{ii+1}; 
% %       case 'savereconerror';      savereconerror      = varargin{ii+1}; 
% %       case 'reconerrortype';      reconerrortype      = varargin{ii+1};
% %       case 'savecputime';         savecputime         = varargin{ii+1}; 
% %       case 'savealphapath';       savealphapath       = varargin{ii+1};
% %       case 'saveiteratepath';     saveiteratepath     = varargin{ii+1}; 
% %       case 'savesteppath';        savesteppath        = varargin{ii+1};
%       % Advanced options
% %       case 'warnings';            warnings            = varargin{ii+1};
%     otherwise
%       % Something wrong with the parameter string
%       error(['Unrecognized option: ''', varargin{ii}, '''']);
%     end
%   end
% end

% If the user specified a lower bound different from the zeros vector,
% set a flag so that we use 'lowerbound' in the subproblem solution.
% simplebound = 1;
% % if any(lowerbound ~= zeros(size(lowerbound)))
% %     simplebound = 0;
% % end


% if monotone == 1
  saveobjective = 1;
% end


% % % 
% % % % Fix A and AT to function handles
% % % if isa(A, 'function_handle') % A is a function call, so AT is required
% % %     if isempty(AT) % AT simply not provided
% % %         error(['Parameter ''AT'' not specified.  Please provide a method ',...
% % %             'to compute A''*x matrix-vector products.']);
% % %     else % AT was provided
% % %         if isa(AT, 'function_handle') % A and AT are function calls
% % %             try dummy = y + A(AT(y));
% % %             catch exception; 
% % %                 error('Size incompatability between ''A'' and ''AT''.');
% % %             end
% % %         else % A is a function call, AT is a matrix        
% % %             try dummy = y + A(AT*y);
% % %             catch exception
% % %                 error('Size incompatability between ''A'' and ''AT''.');
% % %             end
% % %             AT = @(x) AT*x; % Define AT as a function call
% % %         end
% % %     end
% % % else
% % %     if isempty(AT) % A is a matrix, and AT not provided.
% % %         AT = @(x) A'*x; % Just define function calls.
% % %         A = @(x) A*x;
% % %     else % A is a matrix, and AT provided, we need to check
% % %         if isa(AT, 'function_handle') % A is a matrix, AT is a function call            
% % %             try dummy = y + A*AT(y);
% % %             catch exception
% % %                 error('Size incompatability between ''A'' and ''AT''.');
% % %             end
% % %             A = @(x) A*x; % Define A as a function call
% % %         else % A and AT are matrices
% % %             try dummy = y + A*AT*y;
% % %             catch exception
% % %                 error('Size incompatability between ''A'' and ''AT''.');
% % %             end
% % %             AT = @(x) AT*x; % Define A and AT as function calls
% % %             A = @(x) A*x;
% % %         end
% % %     end
% % % end


%============================
%= Initialize the algorithm =
%============================
% Compute an initial point


%xinit = zeros(m2,1);




% Prealocate arrays for storing results
termval = zeros(maxiter+1,1);

% if saveobjective;   objective = zeros(maxiter+1,1);   end
% if savereconerror;  reconerror = zeros(maxiter+1,1);  end
% if savecputime;     cputime = zeros(maxiter+1,1);     end
% if savealphapath;   alphapath = zeros(maxiter+1,1);   end
% if saveiteratepath; iteratepath = cell(maxiter+1,1);  end
% if savesteppath;    steppath = cell(maxiter+1,1);     end
% Precompue initial quantities 
x = xinit;
Ax = A*x;
xprevious = x;
Axprevious = Ax;
dx = x - xprevious;
grad = Aprime*(exp(Ax) - y); 
% if alphamethod ~= 0
%   sqrty = sqrt(y);
% end
% Save related quantities for the initial values
[termval(iter+1), ~] = checkTermination;
 if saveobjective;   objective(iter+1) = computeObjective;    end
% if savereconerror;  reconerror(iter+1) = computeReconError;  end
% if savecputime;     cputime(iter+1) = 0;                     end
% if savealphapath;   alphapath(iter+1) = alpha;               end
% if saveiteratepath; iteratepath{iter+1} = x;                 end
% if savesteppath;    steppath{iter+1} = zeros(size(x));       end


%=============================
%= Display beginning message =
%=============================
if (verbose > 0); thetime = fix(clock);
  for ii = 1:60; fprintf('='); end; fprintf('\n');
  fprintf([ '= Beginning SPIRAL Canonical            ',...
            '@ %2d:%02d %02d/%02d/%4d =\n',...
            '=   Miniter:  %-5d               ',...
            'Maxiter:  %-5d          =\n'],...
    thetime(4),thetime(5),thetime(2),thetime(3),thetime(1),miniter,maxiter);  
  for ii = 1:60; fprintf('='); end; fprintf('\n');
end
% Show initial values:
%jo
% 
% if ~mod(iter,verbose)
%   fprintf('%3d|',iter)
%   if savecputime
%     fprintf(' t:%3.2fs',cputime(iter+1))
%   end
%   fprintf(' init alpha:%10.4e',alpha);
%   if saveobjective
%     fprintf(' init obj:%11.4e',objective(iter+1));
%   end
%   if savereconerror
%     fprintf(' init err:%10.4e', reconerror(iter+1));
%   end
%   fprintf(' init term:%11.4e (target %10.4e)', termval(iter+1),tolerance);
%   fprintf('\n');
% end


% if savecputime; tic; end
iter = iter+1;


%=============================
%= Begin main algorithm loop =
%=============================
while (iter <= miniter) || ((iter <= maxiter) && ~converged)
  
  %=== Compute next iterate ===
% 	switch alphamethod
%     case 0 % Constant alpha throughout all iterations
%       dx = xprevious;
%       step = xprevious - grad./alpha;
% %x = computeSubproblemSolution;
% %jo
%       quotient=-tau./alpha; 
%       x = max( 0, step+ quotient) - max( 0, -step + quotient);     
%       dx = x - dx;
%       Ax = A*x;
%       if saveobjective; objective(iter+1) = computeObjective; end
%       if savealphapath; alphapath(iter+1) = alpha; end
%       
%     case 1 % Barzilai-Borwein choice of alpha
      if monotone % Check acceptance criterion
        past = (max(iter-1-acceptpast,0):iter-1) + 1;
        maxpastobjective = max(objective(past));
        accept = 0;
        while (accept == 0)
          % Compute a candidate next iterate
          dx = xprevious;
          step = xprevious - grad./alpha;
%jo
%           x = computeSubproblemSolution;
          quotient=-tau./alpha; 
          x = max( 0, step+ quotient) - max( 0, -step + quotient); 
          dx = x - dx;
          %Adx = Axprevious;
          Ax = A*x;
          Adx = Ax - Axprevious;
          normsqdx = sum(dx(:).^2);
          
          % Compute the resulting objective 
          objective(iter+1) = computeObjective;
          if ( objective(iter+1) <= (maxpastobjective ...
                - acceptdecrease*alpha/2*normsqdx) ) ...
                || (alpha >= acceptalphamax);
            accept = 1;
          end
          %alphaaccept = alpha;  % Keep value for displaying
          alpha = acceptmult*alpha;
        end
%         if savealphapath; alphapath(iter+1) = alphaaccept; end
      else
        % Just take the Barzilai-Borwein step, no enforcing monotonicity.
        dx = xprevious;
        step = xprevious - grad./alpha;
%jo
%         x = computeSubproblemSolution;
        quotient=-tau./alpha; 
        x = max( 0, step+ quotient) - max( 0, -step + quotient); 
        dx = x - dx;
        Adx = Axprevious;
        Ax = A*x;
        Adx = Ax - Adx;
        normsqdx = sum(dx(:).^2);
%         if saveobjective; objective(iter+1) = computeObjective; end
%         if savealphapath; alphapath(iter+1) = alpha; end
      end
      
      % Update alpha using Barzilai-Borwein choice of alpha
      % Adx is overwritten at top of iteration, so we can reuse it here
%       Adx = Adx.*sqrty./(Ax + epsilon);
%       gamma = sum(Adx(:).^2);
      
      gamma = Adx'*(Adx.*exp(Ax));
%jo       
%        if gamma == 0
%          alpha = alphamin;
%        else
        alpha = min(alphamax, max(gamma/normsqdx, alphamin));
%        end
      
%   end
  
  %=== Compute items for next iteration ===
  xprevious = x;
  Axprevious = Ax; 
  diff_expAx_y=exp(Ax) - y;
  grad = Aprime*(diff_expAx_y); 

  %=== Check convergence and calculate output quantities ===
  [termval(iter+1) converged] = checkTermination;
  % Note: Objective is saved above (since it is needed when monotone = 1)
%   if savereconerror;  reconerror(iter+1) = computeReconError; end
%   if savecputime;     cputime(iter+1) = toc;                  end
%   if saveiteratepath; iteratepath{(iter+1)} = x;              end
%   if savesteppath;    steppath{iter+1} = step;                end
  
  %=== Display Progress ===
  %jo
  %displayProgress;
    
	iter = iter + 1;
end
%===========================
%= End main algorithm loop =
%===========================

%==============================
%= Generate output quantities =
%==============================
% Determine what needs to be saved in the variable output, and crop the output
% if the maximum number of iterations were not performed.
% Note: iter needs to be decremented since it is always incremented in the loop
% varargout = {iter-1};
% varargout = [varargout {termval(1:iter)}];
% if saveobjective;     varargout = [varargout {objective(1:iter)}];    end
% if savereconerror;    varargout = [varargout {reconerror(1:iter)}];   end
% if savecputime;       varargout = [varargout {cputime(1:iter)}];      end
% if savealphapath;     varargout = [varargout {alphapath(1:iter)}];    end
% if saveiteratepath;   varargout = [varargout {iteratepath(1:iter)}];  end
% if savesteppath;      varargout = [varargout {steppath(1:iter)}];     end

%==============================
%= Display completion message =
%==============================
if (verbose > 0); thetime = fix(clock);
  for ii = 1:60; fprintf('='); end; fprintf('\n');
  fprintf([ '= Completed SPIRAL Canonical           ',...
            ' @ %2d:%02d %02d/%02d/%4d =\n',...
            '=   Iterations performed:  %-5d ',...
            '                          =\n'],...
    thetime(4),thetime(5),thetime(2),thetime(3),thetime(1),iter-1);  
  for ii = 1:60; fprintf('='); end; fprintf('\n');
end


%===============================================================================
%= Helper Subfunctions =========================================================
%===============================================================================

%========================
%= Gradient Computation =
%========================
% function grad = computeGradient
% %     grad = AT(1 - (y./(Ax + epsilon)));
%     grad = AT(exp(Ax) - y);      
% end

%=========================
%= Objective Computation =
%=========================
function objective = computeObjective
    precompute = exp(Ax)-y.*Ax;
    objective = sum(precompute)+ tau*sum(abs(x));
end

%==========================
%= Subproblem Computation =
%==========================
%jo
% % % function subsolution = computeSubproblemSolution
% % %         quotient=-tau./alpha; 
% % %         subsolution = max( 0, step+ quotient) - max( 0, -step + quotient);
% % % end

% function y = soft(x,thresh)
% y = max( 0, x - thresh) - max( 0, -x - thresh);
% end


%====================================
%= Reconstruction Error Computation =
%====================================
function reconerror = computeReconError
  errorvect = x - truth;
  switch reconerrortype
    % - Based on squared error -
    case 0 % Squared error
      reconerror = sum(errorvect(:).^2);
    case 1 % Squared error per pixel
      reconerror = sum(errorvect(:).^2)./numel(errorvect);
    case 2 % Relative squared error
      reconerror = sum(errorvect(:).^2)./sum(truth(:).^2);
    case 3 % Relative squared error, as a percent
      reconerror = 100*sum(errorvect(:).^2)./sum(truth(:).^2);
      
    % - Based on l2 norm -
    case 4 % Root squared error (l2 norm)
      reconerror = sqrt(sum(errorvect(:).^2));
    case 5 % Relative root squared error (l2 norm)
      reconerror = sqrt(sum(errorvect(:).^2))./sqrt(sum(truth(:).^2));
    case 6 % Relative root squared error (l2 norm), as a percent
      reconerror = 100*sqrt(sum(errorvect(:).^2))./sqrt(sum(truth(:).^2));
    
    % - Based on l1 norm -
    case 7 % Absolute error (l1 norm)
      reconerror = sum(abs(errorvect(:)));
    case 8 % Absolute error (l1 norm) per pixel
      reconerror = sum(abs(errorvect(:)))./numel(errorvect);
    case 9 % Relative absolute error (l1 norm)
      reconerror = sum(abs(errorvect(:)))./sum(abs(truth(:)));
    case 10 % Relative absolute error (l1 norm), as a percent
      reconerror = 100*sum(abs(errorvect(:)))./sum(abs(truth(:)));
      
    % - Based on PSNR -
    case 11 % PSNR, using maximum true intensity
      reconerror = sum(errorvect(:).^2)./numel(errorvect);
      reconerror = max(truth(:)).^2./reconerror;
      reconerror = 10*log10(reconerror);
    case 12 % PSNR, using dynamic range
      reconerror = sum(errorvect(:).^2)./numel(errorvect);
      reconerror = (max(truth(:)) - min(truth(:))).^2./reconerror;
      reconerror = 10*log10(reconerror);
  end
end

%====================================
%= Termination Criteria Computation =
%====================================
function [termval converged] = checkTermination
      if iter == 0
        termval = NaN;
      else
        termval = (sum(dx(:).^2)./sum(x(:).^2));
      end
      converged = (termval <= tolerance);
    
    
% switch stopcriterion
%     case 0 % Simply exhaust the maximum iteration budget
%     case 1 % Terminate after a specified CPU time (in seconds)
%       termval = cputime(iter+1);
%       converged = (termval >= tolerance);
%       
%     % - Based on changes in iterates -   
%     case 2 % The l2 change in the iterates
%       if iter == 0
%         termval = NaN;
%       else
%         termval = sqrt(sum(dx(:).^2));
%       end
%       converged = (termval <= tolerance);
%     case 3 % The relative l2 change in the 
%       if iter == 0
%         termval = NaN;
%       else
%         termval = (sum(dx(:).^2)./sum(x(:).^2));
%       end
%       converged = (termval <= tolerance);
%       
%     % - Based on changes in objective -   
%     case 4 % The change in the objective (increasing or decreasing)
%       if iter == 0
%         termval = NaN;
%       else
%         termval = abs(objective(iter+1) - objective(iter));
%       end
%       converged = (termval <= tolerance);
%     case 5 % The change in the objective (only decreasing)
%       if iter == 0
%         termval = NaN;
%       else
%         termval = objective(iter+1) - objective(iter);
%       end
%       converged = (abs(termval) <= tolerance) && (termval <= 0);
%     case 6 % The relative change in the objective (increasing or decreasing)
%       if iter == 0
%         termval = NaN;
%       else
%         termval = abs(objective(iter+1) - objective(iter))./abs(objective(iter+1));
%       end
%       converged = (termval <= tolerance);
%     case 7 % The relative change in the objective (only decreasing)
%       if iter == 0
%         termval = NaN;
%       else
%         termval = (objective(iter+1) - objective(iter))./abs(objective(iter+1));
%       end
%       converged = (abs(termval) <= tolerance) && (termval <= 0);
%     
%     % - Based on optimality conditions - 
%     case 8 % Based on KKT conditions
%       termval = abs(sum(x(:).*(grad(:) + tau(:))));
%       converged = (termval <= tolerance);
%       
%   end
end

%====================
%= Display Progress =
%====================
function displayProgress
  if ~mod(iter,verbose)
    fprintf('%3d|',iter)
    if savecputime
      fprintf(' t:%3.2fs',cputime(iter+1))
    end
    fprintf(' dx:%10.4e', sqrt(sum(dx(:))));
    if savealphapath
      fprintf(' alpha used:%10.4e',alphapath(iter+1));
    end
    fprintf(' alpha BB:%10.4e',alpha);
    if saveobjective
      fprintf(' obj:%11.4e',objective(iter+1));
      fprintf(' dobj:%11.4e',objective(iter+1) - objective(iter));
    end
    if savereconerror
      fprintf(' err:%10.4e', reconerror(iter+1));
    end
    fprintf(' term:%11.4e (target %10.4e)', termval(iter+1),tolerance);
    fprintf('\n');
  end
end
    
        
  
end
