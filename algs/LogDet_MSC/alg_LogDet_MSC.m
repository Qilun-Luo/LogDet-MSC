function [C, S, Out] = alg_LogDet_MSC(X, cls_num, gt, opts)

%% Note: Multiview Subspace Clustering with Tensor Log-determinant Model
% Input:
%   X:          data features
%   cls_num:    number of clusters
%   gt:         ground truth clusters
%   opts:       optional parameters
%               - maxIter: max iteration
%               - mu:  penalty parameter
%               - rho: penalty parameter
%               - epsilon: stopping tolerance
% Outout:
%   C:          clusetering results
%   S:          affinity matrix
%   Out:        other output information, e.g. metrics, history

%% Parameter settings
N = size(X{1}, 2);
K = length(X); % number of views

% Default
maxIter = 200;
epsilon = 1e-7;
lambda = 0.2;
mu = 1e-5;
rho = 1e-5;
eta = 2;
max_mu = 1e10; 
max_rho = 1e10;  
flag_debug = 0;

if ~exist('opts', 'var')
    opts = [];
end  
if  isfield(opts, 'maxIter');       maxIter = opts.maxIter;         end
if  isfield(opts, 'epsilon');       epsilon = opts.epsilon;         end
if  isfield(opts, 'lambda');        lambda = opts.lambda;           end
if  isfield(opts, 'mu');            mu = opts.mu;                   end
if  isfield(opts, 'rho');           rho = opts.rho;                 end
if  isfield(opts, 'eta');           eta = opts.eta;                 end
if  isfield(opts, 'max_mu');        max_mu = opts.max_mu;           end
if  isfield(opts, 'max_rho');       max_rho = opts.max_rho;         end
if  isfield(opts, 'flag_debug');    flag_debug = opts.flag_debug;   end


%% Initialize...
for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N); 
    Y{k} = zeros(size(X{k},1),N);   
end

iter = 0;
Isconverg = 0;

%% Iterating
while(Isconverg == 0)
    if flag_debug
        fprintf('----processing iter %d--------\n', iter+1);
    end

    %-------------------Update Z^k-------------------------------    
    for k=1:K
        tmp = X{k}'*Y{k} + mu*X{k}'*X{k} - mu*X{k}'*E{k} - W{k} + rho*G{k};
        Z{k}=(mu*X{k}'*X{k}+rho*eye(N,N))\tmp;
    end
    
    %-------------------Update E^k-------------------------------
    C = [];
    for k=1:K    
        tmp = X{k}-X{k}*Z{k}+Y{k}/mu;
        C = [C; tmp];
    end
    [Econcat] = solve_l1l2(C,lambda/mu);

    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end
  
    %-------------------Update G---------------------------------
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    [G_tensor, objV] = logDet_Shrink(Z_tensor + W_tensor/rho, 1/rho, 3); % Logdet
    
    
    %-------------------Update auxiliary variable---------------
    W_tensor = W_tensor  + rho*(Z_tensor - G_tensor);
    for k=1:K
        Y{k} = Y{k} + mu*(X{k}-X{k}*Z{k}-E{k});
        G{k} = G_tensor(:,:,k);
        W{k} = W_tensor(:,:,k);
    end   
    
    % Record the iteration information
    history.objval(iter+1) = objV;

    % Coverge condition
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epsilon)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
            if flag_debug
                fprintf('    norm_Z     %7.10f    \n', history.norm_Z);
            end
            Isconverg = 0;
        end

        if (norm(Z{k}-G{k},inf)>epsilon)
            history.norm_Z_G = norm(Z{k}-G{k},inf);
            if flag_debug
                fprintf('    norm_Z_G   %7.10f    \n', history.norm_Z_G);
            end
            Isconverg = 0;
        end 
    end
    
    if (iter>maxIter)
        Isconverg  = 1;
    end
    
    % Update penalty params
    mu = min(mu*eta, max_mu);
    rho = min(rho*eta, max_rho);
    
    iter = iter + 1;
end

%% Clustering
S = 0;
for k=1:K
    S = S + abs(Z{k})+abs(Z{k}');
end

C = SpectralClustering(S,cls_num);

[~, nmi, ~] = compute_nmi(gt,C);
ACC = Accuracy(C,double(gt));
[f,p,r] = compute_f(gt,C);
[AR,~,~,~]=RandIndex(gt,C);

%% Record
Out.NMI = nmi;
Out.AR = AR;
Out.ACC = ACC;
Out.recall = r;
Out.precision = p;
Out.fscore = f;

Out.history = history;

