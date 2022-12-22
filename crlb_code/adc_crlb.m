% Calculates the CRLB for ADC as a function of the dependent variables in x, and independent variables b and sigma
function val = adc_crlb(x,b,sigma)
    if size(unique(b),1) > 1  
        df = @(x,b) [exp(-b*x(2)) -b.*x(1).*exp(-b*x(2))]; % partial derivates of ADC model with respect to S_0 (x(1)) and ADC (x(2))
        dy = df(x,b);
        I = (dy'*dy)./sigma^2; % Fisher information matrix (FIM)
        invI = inv(I+eps); % Inverse of the FIM
        val = invI(2,2); % Second diagonal element corresponds to the lower bound on the variance of ADC (x(2)) 
    else
        val = Inf;
    end
end