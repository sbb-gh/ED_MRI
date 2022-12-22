% Calculates the CRLB for T2 multi-echo-spin echo as a function of the dependent variables in x, and independent variables TE and sigma
% x = [S0 R2]
function val = t2mese_crlb(x,te,sigma)
    if size(unique(b),1) > 1
        df = @(x,b) [exp(-te*x(2)) -te.*x(1).*exp(-te*x(2))]; % partial derivates of T2-multi-echo spin echo model with respect to S_0 (x(1)) and R2 (x(2))
        dy = df(x,b);
        I = (dy'*dy)./sigma^2; % Fisher information matrix (FIM)
        invI = inv(I+eps); % Inverse of the FIM
        val = invI(2,2); % Second diagonal element corresponds to the lower bound on the variance of ADC (x(2)) 
    else
        val = Inf;
    end
end