% Calculates the CRLB for T1 inversion recovery function of the dependent variables in x, and independent variables TI and sigma
% x = [S0 R2]
function val = t1invrec_crlb(x,ti,tr,sigma)
    if size(unique(ti),1) > 1
        %df = @(x,ti,tr) [(1 - 2*exp(-ti*x(2)) + exp(-tr*x(2))) x(1)*(2*ti*exp(-ti*x(2)) -tr*exp(-tr*x(2)))]; % partial derivates of T2-multi-echo spin echo model with respect to S_0 (x(1)) and R1 (x(2))
        %dy = df(x,ti,tr);
        dy = [(1 - 2*exp(-ti*x(2)) + exp(-tr*x(2))) x(1)*(2*ti.*exp(-ti*x(2)) -tr*exp(-tr*x(2)))]; 
        I = (dy'*dy)./sigma^2; % Fisher information matrix (FIM)
        invI = inv(I+eps); % Inverse of the FIM
        val = invI(2,2); % Second diagonal element corresponds to the lower bound on the variance of R1 (x(2)) 
    else
        val = Inf;
    end
end