%% 2023.3.1
%% Input/Output explanation of the algorithm
%====Input=================================================================
%  1¡¢FunType£ºtest function type
%  2¡¢FunId£ºFunction sequence number
%  3¡¢MaxFEs£ºthe maximum number of computations of a function
%  4¡¢MaxIter£ºMaximum number of iterations
%  5¡¢SwarmSize£ºpopulation number
%  6¡¢Lb£ºthe lower boundary value of the function
%  7¡¢Ub£ºthe upper boundary of the function
%  8¡¢Functions£ºtest function£¬it's used to compute the value of the function
%  9¡¢Dim£ºalgorithm dimension
%====Output=================================================================
%  1¡¢BestX£ºReturns the optimal position
%  2¡¢BestF£ºReturns the optimal function value
%  3¡¢HisBestF£ºReturns the optimal solution for each generation of the iterative procedure
%  4¡¢ArithmeticName£ºAlgorithm name
%%
function [BestX, BestF, HisBestF, ArithmeticName] = iDE_APAMS(FunType, FunId, MaxFEs, MaxIter, SwarmSize, Lb, Ub, Functions, Dim)

            ArithmeticName='iDE_APAMS';
            rand('seed', sum(100 * clock));
            if length(Lb) == 1
                XRange = [Lb * ones(1, Dim); Ub * ones(1, Dim)];
            else
                XRange = [Lb ; Ub];
            end   
%% ======= Initial parameter setting and population initialization  ===============================================================================   
            max_pop_size = SwarmSize;
            min_pop_size = 4.0;            
            archive=[];
            Qbest = 0.11;    
            Tbest = 0.10;    
            arc_rate = 1.4; 
            memory_size = 5; 
            r= 0.80;
            first_percent = 0.2;
            % archive
            archive.NP = arc_rate * SwarmSize; % the maximum size of the archive
            archive.pop = zeros(0, Dim); % the solutions stored in te archive
            archive.funvalues = zeros(0, 1); % the function value of the archived solutions
            
            % Initializes the policy pool to allocate the probability of each policy
            % At the initial moment, the two variant strategies within each strategy pool are equally distributed
            num_de = 2;   
            probDE1 = 1./num_de .* ones(1, num_de); 
            probDE2 = 1./num_de .* ones(1, num_de); 
            
           % === Initialize the main population
            PositionsOld = repmat(XRange(1, :), SwarmSize, 1) + rand(SwarmSize, Dim) .* (repmat(XRange(2, :) - XRange(1, :), SwarmSize, 1));           
            Positions = PositionsOld; % the old population becomes the current population
            fitness = Functions(FunType,FunId,Positions);
            
            % Update BestX and BestF at the initial time
            t = 0;
            BestF = 1e+100; 
            BestX = zeros(1, Dim);
            HisBestF = NaN(MaxIter, 1);
            for i = 1 : SwarmSize
              t = t + 1;
              if fitness(i) < BestF
                BestF = fitness(i);
                BestX = Positions(i, :);
              end
              if t > MaxFEs; break; 
              end
            end  
            %  the initialization of memory
            memory_sf = 0.5 .* ones(memory_size, 1);
            memory_cr = 0.5 .* ones(memory_size, 1);            
            memory_1st_percent = first_percent.* ones(memory_size, 1);   
            memory_pos = 1;  
            flag = 1;            
           %% Initialize the parameters of the CMA-ES mutation policy
            sigma = 0.5;          % coordinate wise standard deviation (step size)
            xmean = rand(Dim,1);    % objective variables initial point
            mu = SwarmSize / 2;               % number of parents/points for recombination
            weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
            mu = floor(mu);
            weights = weights/sum(weights);     % normalize recombination weights array
            mueff = sum(weights)^2 / sum(weights.^2); % variance-effectiveness of sum w_i x_i            
            % Strategy parameter setting
            cc = (4 + mueff/Dim) / (Dim+4 + 2*mueff/Dim); % time constant for cumulation for C
            cs = (mueff + 2) / (Dim + mueff + 5);  % t-const for cumulation for sigma control
            c1 = 2 / ((Dim + 1.3)^2 +  mueff);    % learning rate for rank-one update of C
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((Dim + 2)^2 + mueff));  % and for rank-mu update
            damps = 1 + 2 * max(0, sqrt((mueff - 1)/(Dim + 1)) - 1) + cs; % damping for sigma usually close to 1
            % Initialize dynamic (internal) strategy parameters and constants
            pc = zeros(Dim,1);
            ps = zeros(Dim,1);                  % evolution paths for C and sigma
            B = eye(Dim,Dim);                   % B defines the coordinate system
            D = ones(Dim,1);                    % diagonal D defines the scaling
            C = B * diag(D.^2) * B';            % covariance matrix C
            invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
            eigeneval = 0;                      % track update of B and D
            chiN = Dim^0.5 * (1 - 1/(4 * Dim) + 1/(21 * Dim^2));  % expectation of            
           %%
            % Number of iterations of the main loop
            Iter = 0;   
            
            % Calculate the fitness ranking at the initial moment
            [~, Sorted_index] = sort(fitness, 'ascend');  
            [~,sorted_fitness] = sort(Sorted_index); 

%% ======== main loop ==========================================================================
      while t < MaxFEs
                Positions = PositionsOld; % the old population becomes the current population

                [~, sorted_index] = sort(fitness, 'ascend');

                mem_rand_index = ceil(memory_size * rand(SwarmSize, 1));
                mu_sf = memory_sf(mem_rand_index);
                mu_cr = memory_cr(mem_rand_index);
                mem_rand_ratio = rand(SwarmSize, 1);
                
               % Counts the individuals assigned to each policy pool
                de_1 = ( memory_1st_percent( mem_rand_index ) >= mem_rand_ratio );
                de_2 = ~de_1;
               % for generating crossover rate
                cr = normrnd(mu_cr, 0.1);
                term_pos = find(mu_cr == -1);
                cr(term_pos) = 0;
                cr = min(cr, 1);
                cr = max(cr, 0);
                
               % for generating scaling factor
                if( t <= MaxFEs / 2 )
                    sf = 0.5 * ones(SwarmSize, 1);
                else
                    sf = mu_sf + 0.1 * tan(pi * (rand(SwarmSize, 1) - 0.5));                   
                    pos = find(sf <= 0);                   
                    while ~ isempty(pos)
                        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                        pos = find(sf <= 0);
                    end
                end
                sf = min(sf, 1);                
               %%                 
                r0 = 1 : SwarmSize;
                popAll = [Positions; archive.pop];
                [r1, r2, r3] = gnR1R2(SwarmSize, size(popAll, 1), r0);
                % Produces the top Qbest% of individuals
                pNP = max(round(Qbest * SwarmSize), 2); %% choose at least two best solutions
                randindex = ceil(rand(1, SwarmSize) .* pNP); %% select from [1, 2, 3, ..., pNP]
                randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
                qbest = Positions(sorted_index(randindex), :); %% randomly choose one of the top 100p% solutions

                tNP = max(round(Tbest * SwarmSize), 2); %% choose at least two best solutions
                tRandindex = ceil(rand(1, SwarmSize) .* tNP); %% select from [1, 2, 3, ..., pNP]
                tRandindex = max(1, tRandindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
                Tbestind = sorted_index(tRandindex);

                % The top Tbest% individuals were ranked according to fitness value
                R = Gen_R(SwarmSize,2);
                R(:,1) = [];
                R = [R Tbestind];
                fr = sorted_fitness(R);
                [~ ,I] = sort(fr,2);
                R_S = [];
                for i = 1:SwarmSize
                    R_S(i,:) = R(i,I(i,:));
                end
                rb = R_S(:,1);
                rm = R_S(:,2);
                rw = R_S(:,3);    
               %%                
                vi = []; % Mutation vector              
                RR = 1 : SwarmSize; % Population number
               %% =========Policy pool 1 :the exploitation policy pool ===================================================================            
                Pos1 = Positions(de_1 == 1,:);    % The populations assigned to strategy pool 1
                size1 = size(Pos1,1);             % The total number of populations assigned to strategy pool 1    
                index1 = RR(de_1 == 1);           % Index of the population assigned to strategy pool 1 in the total population
                rand_index1 = randperm(size1,size1);
                if flag == 1
                    rand_index_1_1 = rand_index1(1:floor(size1 * probDE1(1))); %Assigned to an individual in the development policy pool with mutation policy 1
                    rand_index_1_2 = rand_index1(floor(size1 * probDE1(1))+1 : end);%Assigned to an individual in the development policy pool with mutation policy 2
                else
                    rand_index_1_1 = rand_index1( 1 : end);
                    rand_index_1_2 = [];
                end

                % DE/current-to-ord_tbest/1 (More greedy mutation strategy) to enhance the exploitative ability of the algorithm                              
                vi(index1(rand_index_1_1),:) = Positions(index1(rand_index_1_1),:) + ...
                            sf(index1(rand_index_1_1), ones(1, Dim)) .* (Positions(rb(index1(rand_index_1_1)), :)...
                            - Positions(index1(rand_index_1_1),:) + Positions(rm(index1(rand_index_1_1)), :)...
                            - popAll(rw(index1(rand_index_1_1)), :));

                % CMA_ES mutation strategy
                temp = [];
                for k=1:sum(size(rand_index_1_2,2))
                    temp(:,k) = xmean + sigma * B * (D .* randn(Dim,1)); % m + sig * Normal(0,C)
                end
                vi(index1(rand_index_1_2),:) = temp';

               %% =====Policy pool 2 :the exploration policy pool ==========================================================               
                Pos2 = Positions(de_2 == 1,:);    % The population assigned to strategy pool 2
                size2 = size(Pos2,1);             % The total number of populations assigned to strategy pool 2         
                index2 = RR(de_2 == 1);           % Index of the population assigned to strategy pool 2 in the total population
                rand_index2 = randperm(size2,size2);
                rand_index_2_1 = rand_index2(1:floor(size2 * probDE2(1))); % Assigned to an individual in the exploration policy pool with the mutation strategy 1
                rand_index_2_2 = rand_index2(floor(size2 * probDE2(1))+1 : end);% Assigned to an individual in the exploration policy pool with the mutation strategy 2
                  
                % DE/current-to-qbest with archive/1
                vi(index2(rand_index_2_1), :)  = Positions(index2(rand_index_2_1), :) + ...
                          sf(index2(rand_index_2_1), ones(1, Dim)) .* ...
                          (qbest(index2(rand_index_2_1), :) - Positions(index2(rand_index_2_1), :) +...
                          Positions(r1(index2(rand_index_2_1)), :) - popAll(r2(index2(rand_index_2_1)), :));

                % DE/current-to-rand /1
                vi(index2(rand_index_2_2),:) = Positions(index2(rand_index_2_2),:) + sf(index2(rand_index_2_2), ones(1, Dim))...
                       .* (Positions(r1(index2(rand_index_2_2)), :) - Positions(index2(rand_index_2_2),:) + Positions(r3(index2(rand_index_2_2)), :)...
                       - popAll(r2(index2(rand_index_2_2)), :));
               %% 
                if(~isreal(vi))
                   flag = 0;
                   continue;
                end
              %% Detection boundary condition                               
                vi = boundConstraint(vi, Positions, XRange);
                
              %% General binomial cross               
                mask = rand(SwarmSize, Dim) > cr(:, ones(1, Dim)); % mask is used to indicate which elements of ui comes from the parent
                rows = (1 : SwarmSize)'; cols = floor(rand(SwarmSize, 1) * Dim)+1; % choose one position where the element of ui doesn't come from the parent
                jrand = sub2ind([SwarmSize Dim], rows, cols); mask(jrand) = false;
                ui = vi; ui(mask) = Positions(mask);
              %% Calculate the fitness value of ui
                children_fitness = Functions(FunType,FunId,ui);
                
              %%  Update BestX and BestF               
                for i = 1 : SwarmSize
                    t = t + 1;
                    if children_fitness(i) < BestF
                        BestF = children_fitness(i);
                        BestX = ui(i, :);
                    end
                    if t > MaxFEs; break; end
                end                                
             %% Calculation of infinite norm for optimal position of individual distance
                Distance_now = [];
                for i = 1 : SwarmSize 
                    Distance_now(i) = norm((ui(i,:)- BestX),inf);
                end 
                Distance_now = Distance_now';  
                dif = abs(fitness - children_fitness);
             %% I == 1: the parent is better; I == 2: the offspring is better 
                I = (fitness > children_fitness);
                goodCR = cr(I == 1);
                goodF = sf(I == 1);
                
                dif_val = dif(I == 1);
                % Calculate the fitness improvement of each policy pool
                dif_val_Class_1 = dif(and(I,de_1) == 1);
                dif_val_Class_2 = dif(and(I,de_2) == 1);
                % The population diversity of each strategy pool was calculated
                dis_val_Class_1 = mean(Distance_now(de_1==1));
                dis_val_Class_2 = mean(Distance_now(de_2==1));

                UI = ui(I == 1,:);
                Distance = [];
                for j = 1:size(UI,1)
                    Distance(j) = norm((UI(j,:) - BestX),inf);
                end
                Distance = Distance';
                 
                num_success_params = numel(goodCR);               
                if num_success_params > 0
                    sum_dif = sum(dif_val);
                    dif_val_f = dif_val / sum_dif;                    
                    dif_val_d = Distance / sum(Distance);
                    w1 = 0.8 * exp(6 * (1-Iter) / MaxIter);
                    % The weight update strategy uses a combination of distance-based and fitness-based (convex combination) to adaptively update CR and F parameters             
                    fin_weight = (1-w1) * dif_val_f + w1 * dif_val_d;                    
                    % for updating the memory of scaling factor
                    memory_sf(memory_pos) = (fin_weight' * (goodF .^ 2)) / (fin_weight' * goodF);                    
                    % for updating the memory of crossover rate
                    if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                        memory_cr(memory_pos)  = -1;
                    else
                        memory_cr(memory_pos) = (fin_weight' * (goodCR .^ 2)) / (fin_weight' * goodCR);
                    end                      
                    % Update the historical memory of population allocation
                    fit_improve = (mean(dif_val_Class_1) / (mean(dif_val_Class_1)+ mean(dif_val_Class_2)));
                    dis_improve = (sum(dis_val_Class_1) / (sum(dis_val_Class_1)+ sum(dis_val_Class_2)));
                    fin_improve = (1-w1) * fit_improve + w1 * dis_improve;
                    memory_1st_percent(memory_pos) = memory_1st_percent(memory_pos) * r+(1 - r) * fin_improve;
                    memory_1st_percent(memory_pos) = min(memory_1st_percent(memory_pos),0.8);
                    memory_1st_percent(memory_pos) = max(memory_1st_percent(memory_pos),0.2);                    
                    memory_pos = memory_pos + 1;
                    if memory_pos > memory_size;  memory_pos = 1; end
                end
                
                %% ======================================================
                  diff2_1 = max(0,(fitness - children_fitness));
                  % Population allocation within the exploitation strategy pool (based on fitness improvement)
                  meanDif_1_1= mean(diff2_1(index1(rand_index_1_1)))/(mean(diff2_1(index1(rand_index_1_1))) + mean(diff2_1(index1(rand_index_1_2))));
                  meanDif_1_2= mean(diff2_1(index1(rand_index_1_2)))/(mean(diff2_1(index1(rand_index_1_1))) + mean(diff2_1(index1(rand_index_1_2))));                                                    
                  count_S_1(1)=max(0,meanDif_1_1);
                  count_S_1(2)=max(0,meanDif_1_2);
                  if count_S_1(1) == 0 && count_S_1(2) == 0
                      probDE1 = 1.0/2 * ones(1,2);
                  else
                      probDE1 = max(0.1,min(0.9,count_S_1./(sum(count_S_1))));
                  end 
                  
                  % Population allocation within the exploration policy pool (based on population diversity)                 
                  meanDis_2_1 = mean(Distance_now(index2(rand_index_2_1)))/mean(Distance_now(index2(rand_index_2_1))) + mean(Distance_now(index2(rand_index_2_2)));
                  meanDis_2_2 = mean(Distance_now(index2(rand_index_2_2)))/mean(Distance_now(index2(rand_index_2_1))) + mean(Distance_now(index2(rand_index_2_2)));                
                  count_S_2(1)=max(0, meanDis_2_1);
                  count_S_2(2)=max(0, meanDis_2_2);                                                  
                  if count_S_2(1) == 0 && count_S_2(2) == 0
                      probDE2 = 1.0/2 * ones(1,2);
                  else
                      probDE2 = max(0.1,min(0.9,count_S_2./(sum(count_S_2))));
                  end                  
                 % Update of archive
                 archive = updateArchive(archive, PositionsOld(I == 1, :), fitness(I == 1));
                %%  Selection strategy
                 [fitness, I] = min([fitness, children_fitness], [], 2);                
                 PositionsOld = Positions;
                 PositionsOld(I == 2, :) = ui(I == 2, :);    
                 
                % Levy flight disturbance was performed on the top QBest% individuals
                if t >= MaxFEs/2
                        [~, sorted_index] = sort(fitness, 'ascend');
                        PNP = max(round(Qbest * SwarmSize), 2); 
                        Preindex = 1:PNP;
                        Levy_PoP = Positions(sorted_index(Preindex), :);
                    for g = 1:PNP
                        New_Levy_PoP(g,:)=Levy(Levy_PoP(g,:),BestX,Lb,Ub,w1,Dim);
                        New_Levy_Fitness(g) = Functions(FunType,FunId,New_Levy_PoP(g,:));
                        if New_Levy_Fitness(g) < BestF
                            BestF = New_Levy_Fitness(g);
                            BestX = New_Levy_PoP(g,:);
                        end
                        if New_Levy_Fitness(g) < fitness(sorted_index(g))
                            PositionsOld(sorted_index(g),:)= New_Levy_PoP(g,:);
                            fitness(sorted_index(g)) = New_Levy_Fitness(g);
                        end
                    end
                end
              %%
               % for resizing the population size
                plan_pop_size = round((((min_pop_size - max_pop_size) / MaxFEs) * t) + max_pop_size);
                
                if SwarmSize > plan_pop_size
                    reduction_ind_num = SwarmSize - plan_pop_size;
                    if SwarmSize - reduction_ind_num <  min_pop_size; reduction_ind_num = SwarmSize - min_pop_size;end
                    
                    SwarmSize = SwarmSize - reduction_ind_num;
                    for rr = 1 : reduction_ind_num
                        [~, indBest] = sort(fitness, 'ascend');
                        worst_ind = indBest(end);
                        PositionsOld(worst_ind,:) = [];
                        Positions(worst_ind,:) = [];
                        fitness(worst_ind,:) = [];
                        I(worst_ind,:) = [];
                    end
                                       
                    archive.NP = round(arc_rate * SwarmSize);                    
                    if size(archive.pop, 1) > archive.NP
                        rndpos = randperm(size(archive.pop, 1));
                        rndpos = rndpos(1 : archive.NP);
                        archive.pop = archive.pop(rndpos, :);
                    end                    
                    % Update the parameters of the CMA_ES mutation strategy
                    mu = SwarmSize/2;               % number of parents/points for recombination
                    weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
                    mu = floor(mu);
                    weights = weights/sum(weights);     % normalize recombination weights array
                    mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
                end
               %%  Update the parameters of the CMA_ES mutation strategy
                if(flag == 1)
                    % Sort by fitness and compute weighted mean into xmean
                    [~, popindex] = sort(fitness);  % minimization
                    xold = xmean;
                    xmean = PositionsOld(popindex(1:mu),:)' * weights;  % recombination, new mean value
                    
                    % Cumulation: Update evolution paths
                    ps = (1-cs) * ps ...
                        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
                    hsig = sum(ps.^2)/(1-(1-cs)^(2*t/SwarmSize))/Dim < 2 + 4/(Dim+1);
                    pc = (1-cc) * pc ...
                        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;                   
                    % Adapt covariance matrix C
                    artmp = (1/sigma) * (PositionsOld(popindex(1:mu),:)' - repmat(xold,1,mu));  % mu difference vectors
                    C = (1-c1-cmu) * C ...                   % regard old matrix
                        + c1 * (pc * pc' ...                % plus rank one update
                        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
                        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update                    
                    % Adapt step size sigma
                    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));                    
                    % Update B and D from C
                    if t - eigeneval > SwarmSize/(c1+cmu)/Dim/10  % to achieve O(problem_size^2)
                        eigeneval = t;
                        C = triu(C) + triu(C,1)'; % enforce symmetry
                        if(sum(sum(isnan(C)))>0 || sum(sum(~isfinite(C)))>0 || ~isreal(C))
                            flag=0;
                            continue;
                        end
                        [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors,find the eigenvalues, the eigenvectors
                        D = sqrt(diag(D));        % D contains standard deviations now
                        invsqrtC = B * diag(D.^-1) * B';
                    end
                end 
               %% Record the optimal solution to the iterative process
                Iter = Iter+1;
                if Iter < MaxIter
                    HisBestF(Iter) = BestF;
                else
                    HisBestF(MaxIter) = BestF;
                end   
               %%  Updated fitness ranking 
                % Fitness ranking
                [~, Sorted_index] = sort(fitness, 'ascend');
                [~,sorted_fitness] = sort(Sorted_index);                   
     end
end
%% 
function vi = boundConstraint (vi, pop, lu)

% if the boundary constraint is violated, set the value to be the middle
% of the previous value and the bound
%
% Version: 1.1   Date: 11/20/2007
% Written by Jingqiao Zhang, jingqiao@gmail.com

[NP, D] = size(pop);  % the population size and the problem's dimension

%% check the lower bound
xl = repmat(lu(1, :), NP, 1);
pos = vi < xl;
vi(pos) = (pop(pos) + xl(pos)) / 2;

%% check the upper bound
xu = repmat(lu(2, :), NP, 1);
pos = vi > xu;
vi(pos) = (pop(pos) + xu(pos)) / 2;
end
%%
function [r1, r2, r3] = gnR1R2(NP1, NP2, r0)

% gnA1A2 generate two column vectors r1 and r2 of size NP1 & NP2, respectively
%    r1's elements are choosen from {1, 2, ..., NP1} & r1(i) ~= r0(i)
%    r2's elements are choosen from {1, 2, ..., NP2} & r2(i) ~= r1(i) & r2(i) ~= r0(i)
%
% Call:
%    [r1 r2 ...] = gnA1A2(NP1)   % r0 is set to be (1:NP1)'
%    [r1 r2 ...] = gnA1A2(NP1, r0) % r0 should be of length NP1
%
% Version: 2.1  Date: 2008/07/01
% Written by Jingqiao Zhang (jingqiao@gmail.com)

NP0 = length(r0);

r1 = floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = (r1 == r0);
    if sum(pos) == 0
        break;
    else % regenerate r1 if it is equal to r0
        r1(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000
        error('Can not genrate r1 in 1000 iterations');
    end
end

r2 = floor(rand(1, NP0) * NP2) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r2 == r1) | (r2 == r0));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
        r2(pos) = floor(rand(1, sum(pos)) * NP2) + 1;
    end
    if i > 1000
        error('Can not genrate r2 in 1000 iterations');
    end
end

r3= floor(rand(1, NP0) * NP1) + 1;
%for i = 1 : inf
for i = 1 : 99999999
    pos = ((r3 == r0) | (r3 == r1) | (r3==r2));
    if sum(pos)==0
        break;
    else % regenerate r2 if it is equal to r0 or r1
         r3(pos) = floor(rand(1, sum(pos)) * NP1) + 1;
    end
    if i > 1000
        error('Can not genrate r3 in 1000 iterations');
    end
end
end
%% 
function R = Gen_R(NP_Size,N)

% Gen_R generate N column vectors r1, r2, ..., rN of size NP_Size
%    R's elements are choosen from {1, 2, ..., NP_Size} & R(j,i) are unique per row

% Call:
%    [R] = Gen_R(NP_Size)   % N is set to be 1;
%    [R] = Gen_R(NP_Size,N) 
%
% Version: 0.1  Date: 2018/02/01
% Written by Anas A. Hadi (anas1401@gmail.com)


R(1,:)=1:NP_Size;

for i=2:N+1
    
    R(i,:) = ceil(rand(NP_Size,1) * NP_Size);
    
    flag=0;
    while flag ~= 1
        pos = (R(i,:) == R(1,:));
        for w=2:i-1
            pos=or(pos,(R(i,:) == R(w,:)));
        end
        if sum(pos) == 0
            flag=1;
        else
            R(i,pos)= floor(rand(sum(pos),1 ) * NP_Size) + 1;
        end
    end
end

R=R';

end
%%
function archive = updateArchive(archive, pop, funvalue)
% Update the archive with input solutions
%   Step 1: Add new solution to the archive
%   Step 2: Remove duplicate elements
%   Step 3: If necessary, randomly remove some solutions to maintain the archive size
%
% Version: 1.1   Date: 2008/04/02
% Written by Jingqiao Zhang (jingqiao@gmail.com)

if archive.NP == 0, return; end

if size(pop, 1) ~= size(funvalue,1), error('check it'); end

% Method 2: Remove duplicate elements
popAll = [archive.pop; pop ];
funvalues = [archive.funvalues; funvalue ];
[~, IX]= unique(popAll, 'rows');
if length(IX) < size(popAll, 1) % There exist some duplicate solutions
  popAll = popAll(IX, :);
  funvalues = funvalues(IX, :);
end

if size(popAll, 1) <= archive.NP   % add all new individuals
  archive.pop = popAll;
  archive.funvalues = funvalues;
else                % randomly remove some solutions
  rndpos = randperm(size(popAll, 1)); % equivelent to "randperm";
  rndpos = rndpos(1 : archive.NP);
  
  archive.pop = popAll  (rndpos, :);
  archive.funvalues = funvalues(rndpos, :);
end
end
%% Levy
function new_Positions = Levy(Positions, Leader_pos, Lb, Ub, StepParam ,dim)
    n = size(Positions, 1);
    beta = 3/2;
%     StepParam = 1.5;
%     beta = 0.8;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    for j = 1:n
        s = Positions(j, :);
        u = randn(size(s)) * sigma;
        v = randn(size(s));
        step = u./abs(v) .^ (1/beta);

        stepsize = StepParam.*step.* (Leader_pos - s);
%         stepsize = step.* (Leader_pos - s);

        s = s + stepsize ;
        new_Positions(j,:) = simplebounds(s, Lb, Ub, dim);
    end
end

% Application of simple constraints
function s = simplebounds(s, Lb, Ub, dim)
    if size(Ub,2) == 1
        for i = 1:dim
            if s(i) < Lb
               s(i) = Lb;
            end

            if s(i) > Ub
               s(i) = Ub;
            end
        end
    else
        for i = 1:dim
            if s(i) < Lb(i)
               s(i) = Lb(i);
            end

            if s(i) > Ub(i)
               s(i) = Ub(i);
            end
        end
    end
end