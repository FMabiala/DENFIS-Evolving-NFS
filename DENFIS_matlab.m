
classdef DENFISS < handle
    % DENFIS - Dynamic Evolving Neuro-Fuzzy Inference System
    % Online evolving fuzzy system with TSK (Takagi-Sugeno-Kang) consequents
    % Uses clustering in [Irradiance, Temperature] input space + local linear models
    
    properties (SetAccess = private)
        % --------------------- +++Hyperparameters +++---------------------
        maxRules     = 60;    % Maximum number of fuzzy rules allowed
        theta        = 0.28;  % Distance threshold for creating a new rule
        K            = 4;     % Number of closest rules used in prediction
        learningRate = 0.06;  % learning Rate (step size) for parameter adaptation
        initSigma    = 0.35;  % Initial width of Gaussian membership functions
        
        % --------------------- Rule Structure ----------------------
        centers      % (maxRules × 2) --> Rule centers [Irradiance, Temperature]
        sigma        % (1 × maxRules) --> Width of Gaussian (clamped ≥ 0.08)
        tskParams    % (maxRules × 3) --> TSK linear coefficients [p0, pG, pT]
        ruleAge      % (1 × maxRules) --> Reserved for future use (e.g. pruning)
        
        nRules       = 0;     % Initial number of active rules
    end
    
    %% =================+++DENFIS METHOD+++===================
    methods
        %% Constructor
        function obj = DENFISS(varargin)
            % Arguments: maxRules, theta, K, learningRate, initSigma
            if nargin >= 1, obj.maxRules     = varargin{1}; end
            if nargin >= 2, obj.theta        = varargin{2}; end
            if nargin >= 3, obj.K            = varargin{3}; end
            if nargin >= 4, obj.learningRate = varargin{4}; end
            if nargin >= 5, obj.initSigma    = varargin{5}; end
            
            % Pre-allocate storage
            obj.centers   = zeros(obj.maxRules, 2);
            obj.sigma     = zeros(1, obj.maxRules);
            obj.tskParams = zeros(obj.maxRules, 3);
            obj.ruleAge   = zeros(1, obj.maxRules);
        end
        
        % ---------------------------------------------------------------==---------------------
        %%  Initial Prediction using TSK current rule base
        % -----------------------------------------------------------------==----------------------
        %  Input:  x = [G, T] row vector
        %  Output: predicted value (clamped to safe range)
        function yPred = predict(obj, x)
            x = x(:).';  % Force row vector [1×2]
            
            % If no rules exist (should never happen after first training)
            if obj.nRules == 0
                yPred = 0;  % 38.0; Safe default fallback
                return;
            end
            
            %% ====+++Step 1: Compute Euclidean Distance to all rule centers+++==== 
            diffMatrix = obj.centers(1:obj.nRules, :) - x;           % (nRules × 2)
            distances  = sqrt(sum(diffMatrix.^2, 2));                % (nRules × 1)
            
            %% ===========+++Step 2: Select K nearest rules +++===============
            kNearest = min(obj.K, obj.nRules);
            [~, sortedIdx] = sort(distances);
            nearestRuleIdx = sortedIdx(1:kNearest);
            
            activation = zeros(1, kNearest);    % Firing strength of each selected rule
            localOutput = zeros(1, kNearest); % Local linear model output
            
            for i = 1:kNearest
                ruleID = nearestRuleIdx(i);
                c = obj.centers(ruleID, :);                     % Center [G T]
                s = max(obj.sigma(ruleID), 0.08);        % Clamped sigma (avoid division by zero)
                p = obj.tskParams(ruleID, :);                % [p0 pG pT]
                
                %% ==============+++GAUSSION MF+++===============
                % Gaussian membership (numerically stable)
                activation(i) = exp(-sum((x - c).^2) / (2*s^2 + 1e-12));
                
                %% ==+++TSK Conseq: y = w0 + wG*Irradiance + wT*Temperature+++==              
                localOutput(i) = p(1) + p(2)*x(1) + p(3)*x(2);
            end
            
            % ========+++Step 3: Weighted average outputs +++===========
            totalActivation = sum(activation);
            if totalActivation > 1e-8
                yPred = sum(activation .* localOutput) / totalActivation;
            else
                yPred = mean(localOutput);  % Degenerate case fallback
            end
            
            %  Final safety clamp: eliminate Inf/NaN and enforce physical bounds 
            % (This is critical for real-time systems where output must stay valid)
            if ~isfinite(yPred) || yPred <= 0 || yPred > 200
                yPred = 0; %38.0;   Reasonable default cap
            end
        end
        
        %% --------------------------------------------------------------==---------------------------------------------
        %%  Online training on a single sample (x, target): Evolves structure + adapts params
        %% -----------------------------------------------------------------==----------------------------------------------
        function yPred = train(obj, x, target)
            x = x(:).';
            target = max(target, 0.1);  % Prevent training with zero/negative targets
            
            %% ---- Case 1: First data point ever → create initial rule 
            if obj.nRules == 0
                obj.nRules = 1;
                obj.centers(1, :)   = x;
                obj.sigma(1)        = obj.initSigma;
                obj.tskParams(1, :) = [target, 0, 0];  % Constant model initially
                yPred = target;
                return;
            end
            
            %% ---- Compute distance to all existing rule centers 
            diffMatrix = obj.centers(1:obj.nRules, :) - x;
            distances  = sqrt(sum(diffMatrix.^2, 2));
            [minDist, winnerIdx] = min(distances);
            
            % ---- Case 2: Input is too far from all rules --> create new rule 
            if minDist > obj.theta && obj.nRules < obj.maxRules
                obj.nRules = obj.nRules + 1;
                newIdx = obj.nRules;
                obj.centers(newIdx, :)   = x;
                obj.sigma(newIdx)        = obj.initSigma;
                obj.tskParams(newIdx, :) = [target, 0, 0];
                yPred = target;
                return;
            end
            
            % ---- Case 3: Adapt the winning (closest) rule 
            winCenter = obj.centers(winnerIdx, :);
            winSigma  = max(obj.sigma(winnerIdx), 0.08);  % Enforce minimum width
            winParams = obj.tskParams(winnerIdx, :);      % [w0 wG wT]
            
            % --- 1. Move center toward current input (simple exponential moving) ---
            obj.centers(winnerIdx, :) = winCenter + obj.learningRate * (x - winCenter);
            
            % --- 2. Adapt Gaussian width based on local data spread
            localSpread = mean(abs(x - obj.centers(winnerIdx, :)));
            obj.sigma(winnerIdx) = max(winSigma + obj.learningRate*(localSpread - winSigma), 0.08);
            
            % --- 3. Local gradient descent on TSK consequent parameters 
            localPrediction = winParams(1) + winParams(2)*x(1) + winParams(3)*x(2);
            
            error = target - localPrediction;
            
            gradient = [1, x(1), x(2)];  % ∂y/∂p for linear model
            obj.tskParams(winnerIdx, :) = winParams + obj.learningRate * error * gradient;
            
            % --- Final output using full ensemble (important for stability)
            yPred = obj.predict(x);
        end
    end
end