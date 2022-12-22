options=optimoptions('fmincon','OptimalityTolerance',1e-12,'StepTolerance',1e-20,'Display','none');%,'MaxFunctionEvaluations',inf,'MaxIterations',3000);
sigma=1; % can be set to any value, as it does not affect the optimum

% Calculate optimal b-value for range of ADCs
adcs = 0.25:0.25:3;
%bmaxes = 0.01:0.01:3;
%figure; hold all;
b_opt = zeros(size(adcs));
for i = 1:size(adcs,2)
    %c = zeros(size(bmaxes));
    %for j = 1:size(bmaxes,2)
    %    c(j) = adc_crlb([1;adcs(i)],[0; bmaxes(j)],sigma);
    %end
    [b_opt(i), val_opt] = fmincon(@(b)adc_crlb([1;adcs(i)],[0; b],sigma),1/adcs(i),[],[],[],[],0,[],[],options);
    %h = plot(bmaxes,sqrt(val_opt)./sqrt(c),'LineWidth',2); line_color = get(h,'Color');
    %h = scatter(max(b_opt),1,50,line_color,'filled'); set(h,'HandleVisibility','off');
end
%xlabel(['b (ms/μm^2)']); ylabel('SNR relative to optimum');
%legend(arrayfun(@(x) ['ADC = ' num2str(x) ' μm^2/ms'],adcs,'UniformOutput',false))

%save the optimised b-values
save("crlb_adc_optimised_protocol.mat","b_opt")





